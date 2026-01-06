# llm-inference-serving-vllm

# Background & Motivation
Why LLM Serving Requires Specialized Infrastructure
Large Language Models are not designed to be invoked as one-off functions in production settings. When deployed for real-world use, an LLM must run as a long-lived service that remains resident in GPU memory and handles a continuous stream of concurrent requests. This introduces systems-level challenges such as **GPU scheduling****,** **latency–throughput trade-offs, and efficient resource sharing across users**. As a result, serving an LLM is fundamentally a systems engineering problem rather than a pure machine learning task.
# The Core Bottleneck — KV Cache Management
During autoregressive decoding, LLMs store Key–Value (KV) tensors for all previously generated tokens to compute attention efficiently. Under concurrent workloads, this KV cache grows rapidly and leads to GPU memory fragmentation, poor batching efficiency, and severe tail-latency degradation when using naive inference approaches. Traditional static batching fails because requests vary in prompt length and generation time. vLLM addresses these issues through PagedAttention, which manages KV cache memory using fixed-size pages, and continuous batching, which allows requests to dynamically enter and exit the decoding process while keeping the GPU fully utilized.
