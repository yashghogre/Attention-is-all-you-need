'''
Importing Libraries
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
from torcheval.metrics.functional import bleu_score
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import math
import random

'''
Model Hyper-parameters
'''
n_emb = 256
vocab_size = 1000
seq_len = 64
batch_size = 64
num_heads = 4
n_dropout = 0.1
torch.manual_seed(1111)
ffwd_w = 1024
num_sa_blocks = 4
num_ca_blocks = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

'''
Tokenization
'''

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

trainer = trainers.BpeTrainer(
        vocab_size = vocab_size,
        special_tokens = ["[START]", "[END]", "[PAD]"]
        )

tokenizer.train(files = ["/kaggle/input/trans-data/source_train.txt", "/kaggle/input/trans-data/target_train.txt"], trainer=trainer)

tokenizer.save("tokenizer.json")

'''
Opening Files and Embedding Vector
'''

with open('/kaggle/input/trans-data/source_train.txt') as f:
    trn_eng = f.readlines()

trn_eng = [l[:-1] for l in trn_eng]
trn_eng = trn_eng[:500000]

with open('/kaggle/input/trans-data/target_train.txt') as f:
    trn_hin = f.readlines()

trn_hin = [l[:-1] for l in trn_hin]
trn_hin = trn_hin[:500000]

with open('/kaggle/input/trans-data/source_test.txt') as f:
    tst_eng = f.readlines()

tst_eng = [l[:-1] for l in tst_eng]

with open('/kaggle/input/trans-data/target_test.txt') as f:
    tst_hin = f.readlines()

tst_hin = [l[:-1] for l in tst_hin]

with open('/kaggle/input/trans-data/source_valid.txt') as f:
    val_eng = f.readlines()

val_eng = [l[:-1] for l in val_eng]

with open('/kaggle/input/trans-data/target_valid.txt') as f:
    val_hin = f.readlines()

val_hin = [l[:-1] for l in val_hin]

embeds = nn.Embedding(vocab_size, n_emb, device=device)

len(trn_eng), len(trn_hin), len(tst_eng), len(tst_hin), len(val_eng), len(val_hin)


def get_tensor(token_id, tokenizer, embeds, pos_encoder, device):
    token_id = torch.tensor(token_id, device=device)
    emb_token = embeds(token_id).to(device)
    emb_token = emb_token.view(1, 1, -1)
    emb_token = pos_encoder(emb_token, seq_len=1)
    return emb_token

def get_tensor_for_inf_x(x, tokenizer, embeds, pos_encoder, seq_len, n_emb, device):
    tokens = tokenizer.encode(x).tokens
    tokens.insert(0, "[START]")
    tokens.append("[END]")

    if len(tokens) > seq_len:
        return "Please make sure the input is not too long"

    while seq_len - len(tokens) > 0:
        tokens.append("[PAD]")

    tokens = [tokenizer.token_to_id(i) for i in tokens]
    t_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    t_t_embs = embeds(t_tokens).to(device)
    
    t_t_embs = t_t_embs.view(1, seq_len, n_emb)
    t_t_embs = pos_encoder(t_t_embs)

    return t_t_embs


'''
Prepare Batch
'''
def get_batch(transformer, split="train", batch_size=16, seq_len=256):
    inp_data = trn_eng if split == "train" else tst_eng
    out_data = trn_hin if split == "train" else tst_hin
    
    # Initialize tensors
    inp_btc = torch.zeros(batch_size, seq_len, transformer.n_emb, device=transformer.device)
    out_btc = torch.zeros(batch_size, seq_len, transformer.n_emb, device=transformer.device)
    target = torch.empty(batch_size, seq_len, dtype=torch.long, device=transformer.device)

    for i in range(batch_size):
        # Get random sample that fits within sequence length
        while True:
            btc_num = random.randint(0, len(inp_data)-1)
            inp = inp_data[btc_num]
            out = out_data[btc_num]

            inp_tokens = transformer.tokenizer.encode(inp).tokens
            out_tokens = transformer.tokenizer.encode(out).tokens

            inp_tokens.insert(0, "[START]")
            inp_tokens.append("[END]")
            out_tokens.insert(0, "[START]")
            out_tokens.append("[END]")

            if len(inp_tokens) <= seq_len and len(out_tokens) <= seq_len:
                break

        # Pad sequences
        while len(inp_tokens) < seq_len:
            inp_tokens.append("[PAD]")
        while len(out_tokens) < seq_len:
            out_tokens.append("[PAD]")

        # Prepare decoder input and target
        dec_inp = out_tokens[:-1]
        target_tokens = out_tokens[1:]

        while len(dec_inp) < seq_len:
            dec_inp.append("[PAD]")
        while len(target_tokens) < seq_len:
            target_tokens.append("[PAD]")

        # Convert tokens to ids
        inp_ids = [transformer.tokenizer.token_to_id(t) for t in inp_tokens]
        out_ids = [transformer.tokenizer.token_to_id(t) for t in dec_inp]
        target_ids = [transformer.tokenizer.token_to_id(t) for t in target_tokens]

        # Convert to tensors
        t_inp = torch.tensor(inp_ids, dtype=torch.long, device=transformer.device)
        t_out = torch.tensor(out_ids, dtype=torch.long, device=transformer.device)
        t_target = torch.tensor(target_ids, dtype=torch.long, device=transformer.device)

        # Store target
        target[i] = t_target

        # Get embeddings
        t_inp_emb = embeds(t_inp)
        t_out_emb = embeds(t_out)

        # Apply positional encoding
        t_inp_emb = t_inp_emb.unsqueeze(0)
        t_out_emb = t_out_emb.unsqueeze(0)
        
        inp_emb = transformer.pos_encoder(t_inp_emb).squeeze(0)
        out_emb = transformer.pos_encoder(t_out_emb).squeeze(0)

        # Store in batch tensors
        inp_btc[i] = inp_emb
        out_btc[i] = out_emb

    return inp_btc, out_btc, target

count = 0
for sen in trn_eng:
    inp_tokens = tokenizer.encode(sen).tokens
    inp_tokens.insert(0, "[START]")
    inp_tokens.append("[END]")

    if len(inp_tokens) <= 64:
        count += 1

print(count)

count = 0
for sen in trn_hin:
    inp_tokens = tokenizer.encode(sen).tokens
    inp_tokens.insert(0, "[START]")
    inp_tokens.append("[END]")

    if len(inp_tokens) <= 64:
        count += 1

print(count)

'''
Positional Embedding
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.device = device
        
        pe = torch.zeros(max_seq_length, d_model, device=device)
        position = torch.arange(0, max_seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        return x + self.pe[:seq_len, :]

'''
Encoder
'''

class SA_Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.wk = nn.Linear(n_emb, head_size, bias=False, device=device)
        self.wq = nn.Linear(n_emb, head_size, bias=False, device=device)
        self.wv = nn.Linear(n_emb, head_size, bias=False, device=device)

    def forward(self, ini_emb):
        k = self.wk(ini_emb)
        q = self.wq(ini_emb)
        v = self.wv(ini_emb)

        k_mul = q @ k.transpose(-2, -1)
        scaling = k_mul * (n_emb**-0.5)
        sm_mul = F.softmax(scaling, dim=-1)
        v_mul = sm_mul @ v

        return v_mul

class SA_MultiHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.head_size = n_emb // num_heads
        self.heads = nn.ModuleList([SA_Head(self.head_size) for _ in range(num_heads)])
        self.lyr = nn.Linear(n_emb, n_emb, bias=False, device=device)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.lyr(out)

        return out


class SA_Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.mh_atn = SA_MultiHead()
        self.dropout = nn.Dropout(n_dropout)
        self.ln1 = nn.LayerNorm(n_emb, device=device)
        self.ln2 = nn.LayerNorm(n_emb, device=device)
        self.ffwd = nn.Sequential(
                nn.Linear(n_emb, ffwd_w, device=device),
                nn.ReLU(),
                nn.Linear(ffwd_w, n_emb, device=device)
                )

    def forward(self, x):
        out = self.ln1(x)
        out = self.mh_atn(out)
        out = self.dropout(out)
        out = out + x
        out1 = self.ln2(out)
        out1 = self.ffwd(out)
        out1 = out1 + out

        return out1


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.sa_blocks_list = []

        for _ in range(num_sa_blocks):
            self.sa_blocks_list.append(SA_Block())

        self.sa_blocks = nn.Sequential(*self.sa_blocks_list)

    def forward(self, x):
        out = self.sa_blocks(x)

        return out

'''
Decoder
'''

class MA_Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.kw = nn.Linear(n_emb, head_size, bias=False, device=device)
        self.qw = nn.Linear(n_emb, head_size, bias=False, device=device)
        self.vw = nn.Linear(n_emb, head_size, bias=False, device=device)

    def forward(self, x):
        k = self.kw(x)
        q = self.qw(x)
        v = self.vw(x)
        mask = torch.triu(torch.full((x.size(1), x.size(1)), float('-inf'), device=device), diagonal=1)
        out = q @ k.transpose(-2, -1)
        out = out * (n_emb ** -0.5)
        out = out + mask
        out = F.softmax(out, dim=-1)
        out = out @ v

        return out

class MA_MultiHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.head_size = n_emb // num_heads
        self.heads = nn.ModuleList([MA_Head(self.head_size) for _ in range(num_heads)])
        self.lyr = nn.Linear(n_emb, n_emb, bias=False, device=device)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.lyr(out)

        return out

class CA_Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.kw = nn.Linear(n_emb, head_size, bias=False, device=device)
        self.qw = nn.Linear(n_emb, head_size, bias=False, device=device)
        self.vw = nn.Linear(n_emb, head_size, bias=False, device=device)

    def forward(self, x_enc, x_dec):
        q = self.qw(x_dec)
        k = self.kw(x_enc)
        v = self.vw(x_enc)

        out = q @ k.transpose(-2, -1)
        out = out * (n_emb ** -0.5)
        out = F.softmax(out, dim=-1)
        out = out @ v

        return out

class CA_MultiHead(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.head_size = n_emb // num_heads
        self.heads = nn.ModuleList([CA_Head(self.head_size) for _ in range(num_heads)])
        self.lyr = nn.Linear(n_emb, n_emb, bias=False, device=device)


    def forward(self, x_enc, x_dec):
        out = torch.cat([h(x_enc, x_dec) for h in self.heads], dim=-1)
        out = self.lyr(out)

        return out

class CA_Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.mmh_attn = MA_MultiHead()
        self.do_1 = nn.Dropout(n_dropout)
        self.cah_attn = CA_MultiHead()
        self.do_2 = nn.Dropout(n_dropout)
        self.ffwd = nn.Sequential(
                nn.Linear(n_emb, ffwd_w, device=device),
                nn.ReLU(),
                nn.Linear(ffwd_w, n_emb, device=device)
                )

        self.ln1 = nn.LayerNorm(n_emb, device=device)
        self.ln2 = nn.LayerNorm(n_emb, device=device)
        self.ln3 = nn.LayerNorm(n_emb, device=device)

    def forward(self, x_enc, x_dec):
        out = self.ln1(x_dec)
        out = self.mmh_attn(out)
        out = self.do_1(out)
        out = out + x_dec
        out1 = self.ln2(out)
        out1 = self.cah_attn(x_enc, out1)
        out1 = self.do_2(out1)
        out1 = out1 + out
        out2 = self.ffwd(out1)
        out2 = out2 + out1
        out2 = self.ln3(out2)

        return out2

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.ca_blocks_list = []

        for _ in range(num_ca_blocks):
            self.ca_blocks_list.append(CA_Block())

        self.ca_blocks = nn.ModuleList(self.ca_blocks_list)

        self.lyr = nn.Linear(n_emb, vocab_size, device=device)

    def forward(self, x_enc, x_dec, split="training"):
        # if purp == "training":
        for block in self.ca_blocks:
            x_dec = block(x_enc, x_dec)

        '''
        Final Layer
        '''
        if split == "training":
          logits = self.lyr(x_dec)
          out = F.softmax(logits, dim=-1)
          out = torch.argmax(out, dim=-1)

        if split == "inference":
          logits = self.lyr(x_dec[:, -1, :])
          out = F.softmax(logits, dim=-1)
          out = torch.argmax(out, -1)

        return logits, out

        # else if purp == "inference":


        # return x_dec

'''
Combined Transformer Class
'''

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_emb, seq_len, device, tokenizer):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(n_emb, seq_len, device)
        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.embeds = nn.Embedding(vocab_size, n_emb, device=device)
        self.tokenizer = tokenizer
        
        self.seq_len = seq_len
        self.n_emb = n_emb
        self.device = device
        
    def forward(self, x_enc, x_dec, targets=None):
        res_enc = self.encoder(x_enc)
        logits, out = self.decoder(res_enc, x_dec)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=2)

        return logits, loss

    def get_batch(self, split="train", batch_size=16):
        return get_batch(self, split, batch_size, self.seq_len)


    def predict(self, x):
        x_dec = get_tensor(
            tokenizer.token_to_id("[START]"), 
            self.tokenizer, 
            embeds, 
            self.pos_encoder, 
            self.device
        )
        
        # Get input embeddings
        inp_embs = get_tensor_for_inf_x(
            x, 
            self.tokenizer, 
            embeds, 
            self.pos_encoder, 
            self.seq_len, 
            self.n_emb, 
            self.device
        )
        
        res_enc = self.encoder(inp_embs)
        final_sentence = []
        max_len = 50
        
        while len(final_sentence) <= max_len:
            logits, out = self.decoder(res_enc, x_dec, split="inference")
            final_token = self.tokenizer.id_to_token(out[-1])
            
            if final_token == "[END]":
                break
                
            final_sentence.append(final_token)
            
            new_embed = get_tensor(
                out[-1],
                self.tokenizer,
                embeds,
                self.pos_encoder,
                self.device
            )
            
            x_dec = torch.cat([x_dec, new_embed], dim=1)

        return " ".join(final_sentence)

def train_step(transformer, optimizer):
    # Get batch
    inp_btc, out_btc, targets = transformer.get_batch("train")
    
    # Forward pass
    logits, loss = transformer(inp_btc, out_btc, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Training loop example
def train_transformer(transformer, lr, epochs=10):
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=1e-6)
    

    for epoch in range(epochs):
        transformer.train()
        train_losses = []

        num_steps = len(trn_eng) // batch_size
        eval_interval = num_steps // 10
        
        for step in range(num_steps):
            # loss = train_step(transformer, optimizer)
            inp_btc, out_btc, targets = transformer.get_batch("train")
        
            # Forward pass
            logits, loss = transformer(inp_btc, out_btc, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            
            if step % eval_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch+1} \t | step: {step} \t | lr: {current_lr} \t | loss: {loss:.4f}")

        transformer.eval()
        test_losses = []
        for _ in range(100):
            with torch.no_grad():
                inp_btc, out_btc, targets = transformer.get_batch("test")
                _, eval_loss = transformer(inp_btc, out_btc, targets)
                test_losses.append(eval_loss.item())

        avg_trn_loss = sum(train_losses) / len(train_losses)  
        avg_tst_loss = sum(test_losses) / len(test_losses)

        print("--------------------------------------")
        print(f"Epoch {epoch+1} summary")
        print(f"Avg Train loss: {avg_trn_loss}")
        print(f"Avg Test loss: {avg_tst_loss}")
        print("--------------------------------------")


transformer = Transformer(
    vocab_size=vocab_size,
    n_emb=n_emb,
    seq_len=seq_len,
    device=device,
    tokenizer=tokenizer
)

trans_params = sum(p.numel() for p in transformer.parameters())
print(f"\n Total Parameters = {trans_params}\n")

# Train the model
train_transformer(transformer, 3e-4, 2)

num = 244
print(val_eng[num])
print(val_hin[num])
print(transformer.predict("hello"))

# num = 100
# bleu_score([transformer.predict(val_eng[num])], [[val_hin[num]]], n_gram=4)