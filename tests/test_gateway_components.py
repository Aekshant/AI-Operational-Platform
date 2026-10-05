import asyncio
import time

import pytest

from forge.gateway.admission import Bulkhead, Shed
from forge.gateway.breaker import CLOSED, HALF_OPEN, OPEN, CircuitBreaker
from forge.gateway.ratelimit import TokenBucket
from forge.gateway.routing import RoutingTable


def test_token_bucket_burst_then_refill():
    b = TokenBucket(rate=10, burst=3)
    assert all(b.try_take()[0] for _ in range(3))
    ok, wait = b.try_take()
    assert not ok and 0 < wait <= 0.1
    time.sleep(0.11)
    assert b.try_take()[0]


def test_breaker_opens_half_opens_and_closes():
    br = CircuitBreaker(threshold=0.5, min_calls=4, open_s=0.05)
    for ok in (True, False, False, False):
        br.record(ok)
    assert br.state == OPEN and not br.allow()
    time.sleep(0.06)
    assert br.allow() and br.state == HALF_OPEN
    assert not br.allow()  # one probe at a time
    br.record(True)
    assert br.state == CLOSED


def test_breaker_failed_probe_reopens():
    br = CircuitBreaker(threshold=0.5, min_calls=2, open_s=0.01)
    br.record(False), br.record(False)
    time.sleep(0.02)
    assert br.allow()
    br.record(False)
    assert br.state == OPEN


async def test_bulkhead_sheds_low_priority_first():
    bh = Bulkhead(max_inflight=1, max_queue=4, queue_timeout_s=1)
    release = asyncio.Event()

    async def hold(prio):
        async with bh.admit(prio):
            await release.wait()

    tasks = [asyncio.create_task(hold(2))]  # takes the only slot
    await asyncio.sleep(0.01)
    tasks += [asyncio.create_task(hold(2)) for _ in range(2)]  # queue = 2 (50% of 4)
    await asyncio.sleep(0.01)
    with pytest.raises(Shed):  # free tier may only use 50% of the queue
        async with bh.admit(0):
            pass
    tasks.append(asyncio.create_task(hold(2)))  # paying tenant still admitted to the queue
    await asyncio.sleep(0.01)
    assert bh.queued == 3
    release.set()
    await asyncio.gather(*tasks)
    assert bh.inflight == 0 and bh.queued == 0


async def test_bulkhead_queue_timeout():
    bh = Bulkhead(max_inflight=1, max_queue=4, queue_timeout_s=0.05)
    async with bh.admit(2):
        with pytest.raises(Shed, match="queue_timeout"):
            async with bh.admit(2):
                pass


def _routes(weight):
    return {"qwen-farm": {"revision": 3, "status": "canary", "backends": [
        {"model": "qwen-farm", "version": "v1", "url": "http://a", "weight": 100 - weight, "role": "stable"},
        {"model": "qwen-farm", "version": "v2", "url": "http://b", "weight": weight, "role": "canary"}],
        "fallback": {"model": "qwen-farm", "version": "v1", "url": "http://a"}}}


def test_routing_split_and_fallback_order():
    rt = RoutingTable("", "k", 1)
    rt.apply(_routes(25))
    picks = [rt.plan("qwen-farm", None, 3)[0].version for _ in range(4000)]
    assert 0.21 < picks.count("v2") / len(picks) < 0.29
    plan = rt.plan("qwen-farm", "session-42", 3)
    assert len(plan) == 3 and plan[1].version != plan[0].version  # retry goes to a different backend first
    assert rt.plan("qwen-farm", "session-42", 3)[0] == plan[0]  # sticky sessions


def test_unknown_deployment():
    rt = RoutingTable("", "k", 1)
    rt.apply(_routes(10))
    assert rt.plan("nope", None, 3) == []
