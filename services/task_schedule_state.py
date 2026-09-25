"""Persist APScheduler next-run times across process restarts."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from core.logger import logger
from services.postgres.db import get_session
from services.postgres.models import AppConfig
from services.task_run_history import latest_finished_run

_TASK_KEYS = {
    "full_sync": "SCHEDULED_TASK_NEXT_RUN_FULL_SYNC",
    "lite_sync": "SCHEDULED_TASK_NEXT_RUN_LITE_SYNC",
    "collections_sync": "SCHEDULED_TASK_NEXT_RUN_COLLECTIONS_SYNC",
}

# Optional "run at this time of day" anchors (HH:MM, server local time / TZ env).
# Empty string = plain interval scheduling (every N hours from the last completion).
_TIME_SETTING_KEYS = {
    "full_sync": "FULL_SYNC_TIME",
    "lite_sync": "LITE_SYNC_TIME",
    "collections_sync": "COLLECTIONS_SYNC_TIME",
}

CATCH_UP_MINUTES = 2
_CATCH_UP_MINUTES = CATCH_UP_MINUTES  # backwards-compatible alias


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time_of_day(raw: object) -> tuple[int, int] | None:
    """Parse ``HH:MM`` (or ``HH:MM:SS``) into ``(hour, minute)``; None when blank/invalid."""
    text = str(raw or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def get_anchor_time(task_key: str) -> tuple[int, int] | None:
    """Configured time-of-day anchor for a task, or None when interval-only."""
    from core.config import settings

    key = _TIME_SETTING_KEYS.get(str(task_key).strip().lower())
    if not key:
        return None
    return parse_time_of_day(getattr(settings, key, ""))


def _to_local_naive(dt: datetime) -> datetime:
    return dt.astimezone().replace(tzinfo=None)


def _from_local_naive(dt: datetime) -> datetime:
    # A naive datetime is interpreted as system local time (honours TZ and DST).
    return dt.astimezone(timezone.utc)


def next_anchored_run(
    anchor: tuple[int, int],
    interval_hours: int,
    after: datetime | None = None,
    *,
    min_gap: timedelta | None = None,
) -> datetime:
    """Next fire time (UTC) on the time-of-day grid strictly after ``after + min_gap``.

    * interval >= 24h: fires once per day-slot at HH:MM (e.g. 168h -> weekly at HH:MM).
    * interval  < 24h: fires at HH:MM and every ``interval`` hours around it within the
      day (e.g. anchor 03:00 + 12h -> 03:00 and 15:00).
    """
    after = after or _utc_now()
    hour, minute = anchor
    safe = max(1, int(interval_hours or 1))
    base = _to_local_naive(after + (min_gap or timedelta(0)))

    if safe >= 24:
        cand = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if cand <= base:
            cand += timedelta(days=1)
        return _from_local_naive(cand)

    step = safe * 60
    first_slot = (hour * 60 + minute) % step
    day0 = base.replace(hour=0, minute=0, second=0, microsecond=0)
    for day in (0, 1):
        day_start = day0 + timedelta(days=day)
        slot = first_slot
        while slot < 24 * 60:
            cand = day_start + timedelta(minutes=slot)
            if cand > base:
                return _from_local_naive(cand)
            slot += step
    return _from_local_naive(day0 + timedelta(days=1, minutes=first_slot))  # pragma: no cover


def is_on_anchor(value: datetime, anchor: tuple[int, int], interval_hours: int) -> bool:
    """True when ``value`` sits on the anchor grid (used to detect a changed time setting)."""
    local = _to_local_naive(value if value.tzinfo else value.replace(tzinfo=timezone.utc))
    if local.second != 0:
        return False
    safe = max(1, int(interval_hours or 1))
    minute_of_day = local.hour * 60 + local.minute
    if safe >= 24:
        return minute_of_day == anchor[0] * 60 + anchor[1]
    step = safe * 60
    return minute_of_day % step == (anchor[0] * 60 + anchor[1]) % step


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def get_persisted_next_run(task_key: str) -> datetime | None:
    key = _TASK_KEYS.get(str(task_key).strip().lower())
    if not key:
        return None
    session = get_session()
    try:
        row = session.query(AppConfig).filter(AppConfig.key == key).first()
        if not row or not row.value:
            return None
        raw = row.value
        if isinstance(raw, dict):
            return _parse_iso(raw.get("next_run"))
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return _parse_iso(payload.get("next_run"))
            except Exception:
                return _parse_iso(raw)
        return None
    finally:
        session.close()


def persist_next_run(task_key: str, next_run: datetime) -> None:
    key = _TASK_KEYS.get(str(task_key).strip().lower())
    if not key:
        return
    if next_run.tzinfo is None:
        next_run = next_run.replace(tzinfo=timezone.utc)
    payload = {"next_run": next_run.astimezone(timezone.utc).isoformat()}
    session = get_session()
    try:
        row = session.query(AppConfig).filter(AppConfig.key == key).first()
        if row:
            row.value = payload
        else:
            session.add(AppConfig(key=key, value=payload))
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.warning("persist_next_run failed task_key=%s: %s", task_key, exc)
    finally:
        session.close()


def is_task_overdue(task_key: str, interval_hours: int, *, now: datetime | None = None) -> bool:
    """True when the schedule is enabled and due.

    A persisted next-run in the past means overdue (do not let a recent last-run
    + interval hide a missed schedule clock). With no persisted next-run, fall
    back to last finished + interval.
    """
    if int(interval_hours or 0) <= 0:
        return False
    when = now or _utc_now()
    safe_hours = max(1, int(interval_hours))
    persisted = get_persisted_next_run(task_key)
    if persisted is not None:
        return persisted <= when

    last = latest_finished_run(task_key)
    if last and last.ended_at:
        ended = last.ended_at
        if ended.tzinfo is None:
            ended = ended.replace(tzinfo=timezone.utc)
        candidate = ended.astimezone(timezone.utc) + timedelta(hours=safe_hours)
        return candidate <= when
    # Enabled schedule with no history and no persisted next-run: treat as due.
    return True


def resolve_next_run_time(task_key: str, interval_hours: int) -> datetime:
    """Next fire time: persisted future time, else last completion + interval, else catch-up soon."""
    now = _utc_now()
    safe_hours = max(1, int(interval_hours or 1))
    key = str(task_key).strip().lower()

    persisted = get_persisted_next_run(key)

    anchor = get_anchor_time(key)
    if anchor is not None:
        if persisted is not None and is_on_anchor(persisted, anchor, safe_hours):
            if persisted > now:
                return persisted
            # Missed while the app was down: catch up soon, then re-anchor after the run.
            catch_up = now + timedelta(minutes=_CATCH_UP_MINUTES)
            persist_next_run(key, catch_up)
            return catch_up
        # No persisted clock, or the time-of-day setting changed: snap to the anchor.
        nxt = next_anchored_run(anchor, safe_hours, now)
        persist_next_run(key, nxt)
        return nxt

    if persisted and persisted > now:
        return persisted
    # Persisted but past: catch up soon (do not rewrite from last-run + interval).
    if persisted is not None and persisted <= now:
        catch_up = now + timedelta(minutes=_CATCH_UP_MINUTES)
        persist_next_run(key, catch_up)
        return catch_up

    last = latest_finished_run(key)
    if last and last.ended_at:
        ended = last.ended_at
        if ended.tzinfo is None:
            ended = ended.replace(tzinfo=timezone.utc)
        candidate = ended.astimezone(timezone.utc) + timedelta(hours=safe_hours)
        if candidate > now:
            persist_next_run(key, candidate)
            return candidate

    catch_up = now + timedelta(minutes=_CATCH_UP_MINUTES)
    persist_next_run(key, catch_up)
    return catch_up


def bump_next_run_after_run(task_key: str, interval_hours: int, *, completed_at: datetime | None = None) -> datetime:
    """After a successful manual or scheduled run, schedule the next interval from completion."""
    ended = completed_at or _utc_now()
    if ended.tzinfo is None:
        ended = ended.replace(tzinfo=timezone.utc)
    safe_hours = max(1, int(interval_hours or 1))
    anchor = get_anchor_time(task_key)
    if anchor is not None:
        # Keep at least ~interval-12h between runs so a manual run at 15:00 does not
        # trigger another one the next morning for a daily/weekly schedule.
        gap = timedelta(hours=safe_hours - 12) if safe_hours >= 24 else timedelta(0)
        nxt = next_anchored_run(anchor, safe_hours, ended, min_gap=gap)
    else:
        nxt = ended.astimezone(timezone.utc) + timedelta(hours=safe_hours)
    persist_next_run(task_key, nxt)
    return nxt
