class GPT2Config:
    def __init__(self, 
                 vocab_size=50257,
                 n_ctx=1024,
                 n_embd=768,
                 n_layer=12,
                 n_head=12,
                 **kwargs):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        
        super().__init__(**kwargs)


    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_dict()}"