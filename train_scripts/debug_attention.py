from diffusers.models.transformers.sana_transformer import *
dim=2240
cross_attention_dim=2240
num_cross_attention_heads=20
cross_attention_head_dim=112
dropout=0.0
attention_out_bias = True
attn = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_cross_attention_heads,
                dim_head=cross_attention_head_dim,
                dropout=0.0,
                bias=True,
                out_bias=attention_out_bias,
                processor=AttnProcessor2_0(),
            )

def print_equal(name,t1,t2):
    print(name,torch.allclose(t1,t2))
device='cuda'
dtype=torch.float64
attn.cuda().to(dtype)
x_len = 11
bsz=2
hidden_states = torch.rand(bsz,1024,cross_attention_dim).to(device=device,dtype=dtype)
encoder_hidden_states = torch.rand(bsz,152,cross_attention_dim).to(device=device,dtype=dtype)
encoder_attention_mask = torch.zeros(bsz,1,152).to(device=device,dtype=dtype)
encoder_attention_mask[:,:,x_len:] = -9984
attn_output = attn(
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=encoder_attention_mask,
    )
encoder_hidden_states_truncated = encoder_hidden_states[:,:x_len]
encoder_attention_mask_truncated = encoder_attention_mask[:,:,:x_len]
attn_output2 = attn(
        hidden_states,
        encoder_hidden_states=encoder_hidden_states_truncated,
        attention_mask=encoder_attention_mask_truncated,
    )
# fws
residual = hidden_states
input_ndim = hidden_states.ndim
batch_size, sequence_length, _ = (
    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
)
batch_size, sequence_length2, _ = (
    hidden_states.shape if encoder_hidden_states_truncated is None else encoder_hidden_states_truncated.shape
)
attention_mask_1 = attn.prepare_attention_mask(encoder_attention_mask, sequence_length, batch_size)
# 40, 1, 152
attention_mask_2 = attn.prepare_attention_mask(encoder_attention_mask_truncated, sequence_length2, batch_size)

print_equal('a', attn_output,attn_output2,)
# res = [
# ('a', attn_output,attn_output2,),
# ('b' ,attention_mask_1[...,:x_len],attention_mask_2)
# ]
# for row in res:
#     print(row[0],torch.allclose(row[1],row[2]))
print_equal('b' ,attention_mask_1[...,:x_len],attention_mask_2)

attention_mask_1 = attention_mask_1.view(batch_size, attn.heads, -1, attention_mask_1.shape[-1])
attention_mask_2 = attention_mask_2.view(batch_size, attn.heads, -1, attention_mask_2.shape[-1])
print_equal('c' ,attention_mask_1[...,:x_len],attention_mask_2)
query = attn.to_q(hidden_states)

key1 = attn.to_k(encoder_hidden_states)
value1 = attn.to_v(encoder_hidden_states) # N L D
   
   
key2 = attn.to_k(encoder_hidden_states_truncated)
value2 = attn.to_v(encoder_hidden_states_truncated)     

key3 = encoder_hidden_states @ attn.to_k.weight
key4 = encoder_hidden_states_truncated @ attn.to_k.weight
print_equal('e' ,key1[:,:x_len],key2)
print_equal('e' ,value1[:,:x_len],value2)
print_equal('e' ,key3[:,:x_len],key4)
inner_dim = key1.shape[-1]
head_dim = inner_dim // attn.heads
        
breakpoint()