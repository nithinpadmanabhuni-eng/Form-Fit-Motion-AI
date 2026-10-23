import streamlit as st
import time
from services.config.workout_config import METRICS_FIELDS
from services.persistence.exercise_repository import add_exercise


def reset_workout_state():
    st.session_state.reps = 0
    st.session_state.current_set_reps = 0
    st.session_state.sets_completed = 0
    st.session_state.workout_completed = False
    st.session_state.last_saved_sets_completed = 0
    st.session_state.last_notified_workout_complete = False
    st.session_state.set_cycle_started_at = time.time()
    st.session_state.last_form_check_ts = 0


def _voice_event(event, exercise, metrics):
    """Send an event to the voice pipeline without ever letting it crash the page.

    An uncaught error here (network, TTS, LLM, ...) stops the Streamlit script
    part-way, so everything rendered after it (the video, history table, ...)
    disappears. Errors are printed to the terminal instead.
    """
    pipeline = st.session_state.get("voice_pipeline")

    if not pipeline:
        return

    try:
        result = pipeline.process_event(
            event=event,
            exercise=exercise,
            metrics=metrics,
        )
    except Exception as e:
        print(f"[voice_pipeline] '{event}' failed: {e}")
        return

    if result:
        st.session_state.audio_to_play, st.session_state.coach_feedback = result


def sync_metrics_update(context):
    # 1) New exercise or new plan (sets / reps) -> start from a clean slate
    plan = (
        st.session_state.get("exercise_type"),
        st.session_state.get("reps_per_set", 0),
        st.session_state.get("target_sets", 0),
    )

    if st.session_state.get("_plan_signature") != plan:
        reset_workout_state()
        st.session_state._plan_signature = plan

    # 2) Camera just started -> the video processor starts counting from 0,
    #    so the stored progress must start from 0 too
    is_playing = bool(context and hasattr(context, "state") and context.state.playing)
    was_playing = st.session_state.get("_was_playing", False)
    st.session_state._was_playing = is_playing

    if is_playing and not was_playing:
        reset_workout_state()

    if not is_playing:
        return

    processor = getattr(context, "video_processor", None)

    if not processor:
        return

    exercise = st.session_state.get("exercise_type")

    if not exercise:
        return

    processor.set_exercise(exercise)
    latest_metrics = processor.get_latest_metrics()

    if not latest_metrics:
        return

    reps = latest_metrics.get("reps", 0)

    if reps is None:
        reps = 0

    st.session_state.reps = reps

    fields = METRICS_FIELDS.get(exercise)

    if not fields:
        return

    for key, default in fields.items():
        st.session_state[key] = latest_metrics.get(key, default)

    reps_per_set = st.session_state.get("reps_per_set", 0)
    target_sets = st.session_state.get("target_sets", 0)

    if reps_per_set > 0 and target_sets > 0:
        # Cap reps at the workout target so extra reps don't roll into a phantom set
        total_target_reps = reps_per_set * target_sets
        capped_reps = min(reps, total_target_reps)

        sets_completed = capped_reps // reps_per_set
        workout_completed = sets_completed >= target_sets

        if workout_completed:
            current_set_reps = reps_per_set
        else:
            current_set_reps = capped_reps % reps_per_set
    else:
        sets_completed = 0
        current_set_reps = 0
        workout_completed = False

    st.session_state.sets_completed = sets_completed
    st.session_state.current_set_reps = current_set_reps
    st.session_state.workout_completed = workout_completed

    last_saved_sets = st.session_state.get("last_saved_sets_completed", 0)

    if target_sets > 0 and reps_per_set > 0 and sets_completed > last_saved_sets:
        newly_completed = sets_completed - last_saved_sets
        now_ts = time.time()
        started_at = st.session_state.get("set_cycle_started_at", now_ts)
        time_taken = now_ts - started_at
        user_id = st.session_state.get("user_id", 0)

        try:
            add_exercise(user_id, exercise, newly_completed * reps_per_set, newly_completed, time_taken)
        except Exception as e:
            print(f"[add_exercise] failed to save progress: {e}")

        _voice_event("set_completed", exercise, latest_metrics)

        st.session_state.set_cycle_started_at = now_ts
        st.session_state.last_saved_sets_completed = sets_completed

    if workout_completed and not st.session_state.get("last_notified_workout_complete", False):
        st.session_state.last_notified_workout_complete = True

        _voice_event("workout_completed", exercise, latest_metrics)

    pose_detected = latest_metrics.get("pose_detected", True)

    if not pose_detected and not workout_completed:
        _voice_event(
            "no_pose_detected",
            exercise,
            {"issue": "No pose detected! Please step into the camera frame."},
        )

    # Form check at most every 4 seconds so it doesn't block every rerun
    now_check = time.time()

    if now_check - st.session_state.get("last_form_check_ts", 0) >= 4:
        st.session_state.last_form_check_ts = now_check

        _voice_event("ongoing_form_check", exercise, latest_metrics)