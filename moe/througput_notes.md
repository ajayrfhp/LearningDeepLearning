LLM inference has two phases, prefill and decode. 

Defining dimensions
- B Batch size (number of independent user requests)
- S sequence length (number of tokens in the prompt)
- D hidden dimension of the model (4096)
- W weight matrix inside model with ${D,D_{out}}$ 

## 1. Prefill phase
Model ingests user prompt all at once
- Input (X) contains embeddings for all S tokens in the prompt
    - Dimension ${B, S, D}$
- Transformation tokens projected through model layers. Y = X . W
    - $(B, S, D) . (D, D_{out}) = {B, S, D_{out}}$
- Output dimension is ${B, S, D_{out}}$
- Compute cost $2 . (B . S . D . D_{out})$
- Memory load $D . D_{out}$
- Arithmetic intesity $2 . B . S$
    - for batch size = 1 and S 1024, aritmetric intensity is 2048 FLOPS per byte loaded
    - H100 are optimal when its 100-200 operation per byte loaded. 
- Weight matrix W is loaded from VRAM into SRAM. GPU uses W to process all tokens across all inputs. Prefill is compute bound. 

## 2. Decode Phase
Model has processed the prompt. Model is now just passing last token generated through network to predict next one. 
This is matrix-vector multiplication at its core. 
- Input (X) is a tiny matrix with 1 new token $B,1,D$
- Token is projected through model layers
    - $(B,1,D) . (D,D_{out}) = (B, 1, D_{out}) $
- Compute cost $2 . (B . D . D_{out})$
- Memory load $D . D_{out} + 4 . B . S . L . D$
    - KV cache grows linearly with token size L. FP 16 is 2 bytes, kv is 2.
- Arithmentic intesity is < $2 . B$
    - For batch size 1 and sequence length = 1, we are down to less than 2 FLOPS per byte loaded. 

- Decoder is memory bottleneck because you are operationg on next tokens, matrix mult is faster. 

## Active concurrent user streams 
Active Concurrent User Streams ($B$) and $\sum N_{i, \text{out}}$

In the formula $\text{Throughput}_{\text{gen}} = \frac{\sum_{i=1}^{B} N_{i, \text{out}}}{\Delta t_{\text{decode}}}$, we are calculating the total token production volume.
- If Batch size is 64, we can go from 2 FLOPS per byte to 128 FLOPS per byte, raising the aritchmetic intensity
- $\sum_{i=1}^{B} N_{i, \text{out}}$ is number of output tokens generated for a specific user stream during a period of time ${\Delta t_{\text{decode}}}$, then aggregated over B batches. 