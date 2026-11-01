from agent_stack.layer2_rag import HashEmbedder
from agent_stack.layer6_cache_routing import ModelRouter, SemanticCache
from agent_stack.layer6_cache_routing.router import LARGE_MODEL, SMALL_MODEL


def test_semantic_cache_hits_on_identical_query():
    cache = SemanticCache(HashEmbedder(), similarity_threshold=0.9)
    cache.put("How many paid leave days do I get?", "18 days a year.", user_scope="u-1")
    hit = cache.lookup("How many paid leave days do I get?", user_scope="u-1")
    assert hit is not None
    assert hit.answer == "18 days a year."


def test_semantic_cache_never_crosses_user_scope():
    cache = SemanticCache(HashEmbedder(), similarity_threshold=0.9)
    cache.put("What's my leave balance?", "You have 12 days left.", user_scope="u-1")
    hit = cache.lookup("What's my leave balance?", user_scope="u-2")
    assert hit is None


def test_semantic_cache_misses_below_threshold():
    cache = SemanticCache(HashEmbedder(), similarity_threshold=0.99)
    cache.put("leave policy", "answer A", user_scope=None)
    hit = cache.lookup("completely unrelated question about parking", user_scope=None)
    assert hit is None


def test_router_sends_short_simple_messages_to_small_model():
    router = ModelRouter()
    decision = router.route("What's the WiFi password?")
    assert decision.model == SMALL_MODEL


def test_router_sends_long_or_complex_messages_to_large_model():
    router = ModelRouter()
    long_message = " ".join(["word"] * 60)
    assert router.route(long_message).model == LARGE_MODEL

    complex_message = "Can you compare our leave policy with last year's and explain why it changed?"
    assert router.route(complex_message).model == LARGE_MODEL
