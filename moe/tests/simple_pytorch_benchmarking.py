import torch
import torch.nn as nn
import wandb

def to_cuda(obj):
    """Recursively or directly moves tensors or models to the active device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return obj.to(device)

class InefficientBatchProcessor(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
    def forward(self, x):
        """
        x: shape (B, D_in)
        """
        x = to_cuda(x)
        B = x.shape[0]
        outputs = []
        
        # INEFFICIENT: Looping over the batch dimension in Python
        for i in range(B):
            # Isolate a single token
            token = x[i].unsqueeze(0) # shape: (1, D_in)
            
            # Process single token
            out_token = self.linear(token) # shape: (1, D_out)
            
            outputs.append(out_token)
            
        # Reassemble the batch
        return to_cuda(torch.cat(outputs, dim=0)) # shape: (B, D_out)

class EfficientBatchProcessor(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        """
        x: shape (B, D_in)
        """
        # EFFICIENT: A single vectorized Matrix Multiplication (GEMM)
        x = to_cuda(x)
        return self.linear(x) # shape: (B, D_out)

def benchmark_batch_processors():
    B = 10000
    D_IN = 1000
    D_OUT = 1000

    efficient_batch_processor = to_cuda(EfficientBatchProcessor(D_IN, D_OUT))
    inefficient_batch_processor = to_cuda(InefficientBatchProcessor(D_IN, D_OUT))
    sample_input = to_cuda(torch.randn(1, D_IN))
    model_input = to_cuda(torch.randn(B, D_IN))
    


    for model in [inefficient_batch_processor, efficient_batch_processor]:
        model_name = f"{model.__class__.__name__}"
        wandb.init(project=f"batch_processor_benchmark", name=model_name, reinit=True)
        # warm-up
        for _ in range(2):
            _ = model.forward((sample_input))

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = model.forward(model_input)
        end_event.record()

        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        throughput = B * 1000.0 / latency_ms
        wandb.log({
            "latency_ms" : latency_ms,
            "throughput" : throughput
        })




if __name__ == "__main__":
    benchmark_batch_processors()