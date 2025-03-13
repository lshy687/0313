import torch

class PLMEncoder:
    """使用预训练语言模型提取文本嵌入的编码器"""
    
    def __init__(self, model, tokenizer):
        """
        初始化PLM编码器
        
        Args:
            model: 预训练语言模型
            tokenizer: 分词器
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.embedding_cache = {}  # 用于缓存已计算的嵌入
        
    def encode(self, text, use_cache=True):
        """
        编码单个文本
        
        Args:
            text: 输入文本
            use_cache: 是否使用缓存
            
        Returns:
            文本的嵌入表示
        """
        # 检查缓存
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            embedding = hidden_states.mean(dim=1).squeeze().detach().cpu()
            
        # 缓存结果
        if use_cache:
            self.embedding_cache[text] = embedding
            
        return embedding
        
    def encode_batch(self, texts, use_cache=True):
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            use_cache: 是否使用缓存
            
        Returns:
            嵌入表示列表
        """
        embeddings = []
        
        # 检查哪些文本需要编码
        texts_to_encode = []
        cached_indices = []
        new_indices = []
        
        for i, text in enumerate(texts):
            if use_cache and text in self.embedding_cache:
                cached_indices.append(i)
            else:
                texts_to_encode.append(text)
                new_indices.append(i)
        
        # 从缓存获取嵌入
        embeddings = [None] * len(texts)
        for i in cached_indices:
            embeddings[i] = self.embedding_cache[texts[i]]
        
        # 如果有需要新编码的文本
        if texts_to_encode:
            # 批量编码
            inputs = self.tokenizer(texts_to_encode, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                batch_embeddings = hidden_states.mean(dim=1).detach().cpu()
            
            # 将结果放入正确的位置并缓存
            for i, idx in enumerate(new_indices):
                embedding = batch_embeddings[i]
                embeddings[idx] = embedding
                if use_cache:
                    self.embedding_cache[texts[i]] = embedding
        
        # 确保所有位置都有嵌入
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                raise RuntimeError(f"编码器未能为文本 {texts[i]} 生成嵌入")
        
        if all(isinstance(e, torch.Tensor) for e in embeddings):
            return torch.stack(embeddings)
        return embeddings
    
    def clear_cache(self):
        """清除嵌入缓存"""
        self.embedding_cache = {}
        
    def get_similarity(self, text1, text2, use_cache=True):
        """
        计算两个文本的余弦相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            use_cache: 是否使用缓存
            
        Returns:
            两个文本的余弦相似度
        """
        emb1 = self.encode(text1, use_cache=use_cache)
        emb2 = self.encode(text2, use_cache=use_cache)
        
        return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))[0]
        
    def get_batch_similarity(self, query_text, texts, use_cache=True):
        """
        计算查询文本与多个文本的相似度
        
        Args:
            query_text: 查询文本
            texts: 文本列表
            use_cache: 是否使用缓存
            
        Returns:
            查询文本与每个文本的相似度列表
        """
        query_emb = self.encode(query_text, use_cache=use_cache)
        texts_emb = self.encode_batch(texts, use_cache=use_cache)
        
        # 计算余弦相似度
        similarities = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0), 
            texts_emb
        )
        
        return similarities 