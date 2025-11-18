import torch
from transformer import Transformer

def main():
    device = torch.device("cpu")
    parameters = {
        "src_vocab_size": 1000,
        "tgt_vocab_size": 1200,
        "batch_size": 2,
        "src_len": 10,
        "tgt_len": 12,
        "d_model": 512,
    }
    model = Transformer(
        src_vocab_size=parameters["src_vocab_size"],
        tgt_vocab_size=parameters["tgt_vocab_size"],
        d_model=parameters["d_model"],
        num_layers=2,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
    ).to(device)
    print(f"Model loaded in: {device} ...")

    src = torch.randint(
        0, parameters["src_vocab_size"],
        (parameters["batch_size"], parameters["src_len"]),
        device=device
    )
    tgt = torch.randint(
        0, parameters["tgt_vocab_size"],
        (parameters["batch_size"], parameters["tgt_len"]),
        device=device
    )
    src_mask = torch.ones(
        (parameters["batch_size"], 1, 1, parameters["src_len"]),
        dtype=torch.bool, device=device
    )

    logits = model(src, tgt, src_mask)

    print("=== RESULTS ===")
    print(f"Shape logits: {logits.shape}")
    print(f"Expected: ({parameters['batch_size']}, {parameters['tgt_len']}, {parameters['tgt_vocab_size']})")

    # To simulate generation step by step:
    print("=== RESULTS STEP BY STEP ===")
    for t in range(1, parameters["tgt_len"] + 1):
        out = model(src, tgt[:, :t])
        print(f"Step {t}: out shape = {out.shape}")

    # To verify that the computational graph has not been broken, no parameter should have a gradient of None
    loss = logits.mean()
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Attention! Param without grad: {name}")

if __name__ == "__main__":
    main()