| Model | Parameters | Trained on TinyStories? | Tokens per character | Loss per token (nats) | Bits per character |
|---|---|---|---|---|---|
| ForgeGPT | 12.2M | Yes | 0.250 | 1.529 | 0.551 |
| GPT-2 small | 124M | No (zero-shot) | 0.242 | 2.404 | 0.838 |

Scored on 1000 TinyStories validation stories (728,583 characters) that fit in ForgeGPT's 256-token context. Lower bits per character is better.
