import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import time
import requests
import re
import threading
from queue import Queue
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
import random
import gc
import numpy as np
from datetime import datetime
import hashlib

# ==================== 增强配置 ====================
class EnhancedConfig:
    def __init__(self):
        # 模型配置
        self.vocab_size = 15000
        self.d_model = 384
        self.n_heads = 6
        self.n_layers = 6
        self.d_ff = 1024
        self.max_seq_len = 512
        self.dropout = 0.1
        self.batch_size = 2
        self.gradient_accumulation = 8
        
        # 训练配置
        self.epochs = 8
        self.learning_rate = 5e-5
        self.warmup_steps = 1000
        
        # 推理配置
        self.thinking_depth = 3  # 思考深度
        self.knowledge_integration = True  # 知识融合
        self.confidence_threshold = 0.7   # 置信度阈值
        
        # 爬虫配置
        self.max_pages_per_topic = 15
        self.request_delay = 1
        self.timeout = 10
        self.retry_count = 3
        
        # 知识图谱配置
        self.min_entity_freq = 2
        self.max_relations = 10000
        
        # 文件路径
        self.data_file = "enhanced_wiki_data.json"
        self.model_file = "enhanced_ai_model.pth"
        self.knowledge_graph_file = "knowledge_graph.json"
        self.learning_log_file = "learning_log.json"

config = EnhancedConfig()

# ==================== Tokenizer 类定义 ====================

class Tokenizer:
    """分词器类"""
    def __init__(self):
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, texts, max_vocab_size=15000):
        """从文本构建词汇表"""
        char_counter = Counter()
        
        for item in texts:
            if isinstance(item, dict):
                text = item['content']
            else:
                text = item
            char_counter.update(text)
        
        # 取最常见的字符
        for char, count in char_counter.most_common(max_vocab_size - 4):
            self.vocab[char] = len(self.vocab)
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        return self.vocab
    
    def encode(self, text):
        """编码文本为token IDs"""
        return [self.vocab.get(char, 1) for char in text]  # 1是<UNK>
    
    def decode(self, tokens):
        """解码token IDs为文本"""
        return ''.join([self.id_to_token.get(token, '<UNK>') for token in tokens 
                       if token not in [0, 1, 2, 3]])

# ==================== 数据集类定义 ====================

class WikiDataset(Dataset):
    """维基百科数据集"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['content']
        
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)

# ==================== 增强的维基百科爬虫 ====================

class EnhancedWikipediaCrawler:
    """增强版维基百科爬虫，支持知识图谱构建"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.collected_data = []
        self.knowledge_graph = defaultdict(list)
        self.entity_freq = Counter()
        
        # 扩展的知识领域
        self.topics = [
            # 计算机科学
            "人工智能", "机器学习", "深度学习", "神经网络", "自然语言处理", 
            "计算机视觉", "数据科学", "大数据", "云计算", "区块链",
            "算法", "数据结构", "编程语言", "软件工程", "操作系统",
            
            # 数学与科学
            "数学", "物理学", "化学", "生物学", "天文学", 
            "统计学", "概率论", "微积分", "线性代数", "量子力学",
            
            # 人文社科
            "历史", "哲学", "心理学", "经济学", "社会学",
            "文学", "艺术", "音乐", "政治学", "法律","爱情",
            
            # 技术与工程
            "电子工程", "机械工程", "土木工程", "航空航天", "机器人技术",
            "物联网", "5G通信", "纳米技术", "生物技术", "可再生能源",
            "python","c++","c语言","c#","java","html","css","JavaScript"
        ]
    
    def extract_entities(self, text, title):
        """从文本中提取实体和关系"""
        entities = []
        relations = []
        
        # 简单的实体提取（实际应用中可以使用NER模型）
        sentences = re.split(r'[。！？]', text)
        for sentence in sentences:
            if len(sentence) < 10:
                continue
                
            # 提取可能的实体（名词短语）
            words = re.findall(r'[\u4e00-\u9fa5]{2,8}', sentence)
            for word in words:
                if len(word) >= 2:
                    self.entity_freq[word] += 1
                    entities.append(word)
            
            # 提取简单关系 (主语，关系，宾语)
            if len(words) >= 3:
                for i in range(len(words) - 2):
                    subject = words[i]
                    relation = words[i+1]
                    obj = words[i+2]
                    
                    if (self.entity_freq[subject] >= config.min_entity_freq and 
                        self.entity_freq[obj] >= config.min_entity_freq):
                        relations.append({
                            'subject': subject,
                            'relation': relation,
                            'object': obj,
                            'source': title
                        })
        
        return entities, relations
    
    def get_page_content(self, title):
        """获取页面内容并提取知识"""
        try:
            content_url = "https://zh.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'prop': 'extracts|links',
                'titles': title,
                'explaintext': True,
                'format': 'json',
                'pllimit': 20  # 获取相关链接
            }
            
            response = self.session.get(content_url, params=params, timeout=config.timeout)
            if response.status_code == 200:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})
                
                for page_id, page_data in pages.items():
                    if 'extract' in page_data:
                        content = self.clean_content(page_data['extract'])
                        
                        # 提取实体和关系
                        entities, relations = self.extract_entities(content, title)
                        
                        # 更新知识图谱
                        for relation in relations:
                            key = f"{relation['subject']}_{relation['relation']}"
                            self.knowledge_graph[key].append({
                                'object': relation['object'],
                                'source': relation['source'],
                                'confidence': min(1.0, self.entity_freq[relation['subject']] / 10)
                            })
                        
                        return {
                            'content': content,
                            'entities': entities,
                            'relations': relations
                        }
        except Exception as e:
            print(f"获取页面 '{title}' 内容失败: {e}")
        return None
    
    def clean_content(self, content):
        """增强的内容清理"""
        # 移除引用标记
        content = re.sub(r'\[\d+\]', '', content)
        # 移除模板
        content = re.sub(r'\{\{.*?\}\}', '', content)
        # 移除文件链接
        content = re.sub(r'\[\[文件:.*?\]\]', '', content)
        # 处理内部链接 [[链接|显示文本]]
        content = re.sub(r'\[\[.*?\|(.*?)\]\]', r'\1', content)
        content = re.sub(r'\[\[(.*?)\]\]', r'\1', content)
        # 移除多余空白
        content = re.sub(r'\s+', ' ', content)
        return content.strip()
    
    def crawl_topic(self, topic):
        """爬取特定主题并构建知识图谱"""
        print(f"开始爬取主题: {topic}")
        
        page_titles = self.get_page_links(topic)
        if not page_titles:
            print(f"  未找到关于 '{topic}' 的页面")
            return
        
        pages_collected = 0
        for title in page_titles:
            if pages_collected >= config.max_pages_per_topic:
                break
                
            result = self.get_page_content(title)
            if result and len(result['content']) > 200:
                self.collected_data.append({
                    'title': title,
                    'content': result['content'],
                    'topic': topic,
                    'entities': result['entities'],
                    'relations': result['relations'],
                    'timestamp': time.time()
                })
                pages_collected += 1
                print(f"  ✓ 已爬取: {title} (实体: {len(result['entities'])}, 关系: {len(result['relations'])})")
            
            time.sleep(config.request_delay)
    
    def get_page_links(self, topic):
        """获取相关页面链接"""
        for attempt in range(config.retry_count):
            try:
                search_url = "https://zh.wikipedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'list': 'search',
                    'srsearch': topic,
                    'format': 'json',
                    'srlimit': config.max_pages_per_topic
                }
                
                response = self.session.get(search_url, params=params, timeout=config.timeout)
                if response.status_code == 200:
                    data = response.json()
                    return [item['title'] for item in data.get('query', {}).get('search', [])]
            except Exception as e:
                print(f"搜索主题 '{topic}' 失败 (尝试 {attempt+1}): {e}")
                time.sleep(2)
        return []
    
    def save_data(self, filename=None):
        """保存数据"""
        if filename is None:
            filename = config.data_file
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.collected_data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存到: {filename}")
    
    def load_data(self, filename=None):
        """加载数据"""
        if filename is None:
            filename = config.data_file
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.collected_data = json.load(f)
            print(f"从 {filename} 加载了 {len(self.collected_data)} 条数据")
        except FileNotFoundError:
            print(f"数据文件 {filename} 不存在")
            self.collected_data = []
    
    def save_knowledge_graph(self):
        """保存知识图谱"""
        # 限制关系数量
        sorted_relations = sorted(self.knowledge_graph.items(), 
                                 key=lambda x: len(x[1]), reverse=True)
        limited_graph = dict(sorted_relations[:config.max_relations])
        
        with open(config.knowledge_graph_file, 'w', encoding='utf-8') as f:
            json.dump(limited_graph, f, ensure_ascii=False, indent=2)
        print(f"知识图谱已保存: {config.knowledge_graph_file} (关系数: {len(limited_graph)})")
    
    def load_knowledge_graph(self):
        """加载知识图谱"""
        try:
            with open(config.knowledge_graph_file, 'r', encoding='utf-8') as f:
                self.knowledge_graph = json.load(f)
            print(f"知识图谱已加载: {len(self.knowledge_graph)} 个关系")
        except FileNotFoundError:
            print("知识图谱文件不存在")

# ==================== 深度思考AI模型 ====================

class DepthThinkingAttention(nn.Module):
    """深度思考注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(DepthThinkingAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 多套注意力权重，支持多轮思考
        self.w_q = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.w_k = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.w_v = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # 思考门控
        self.thinking_gate = nn.Linear(d_model * 2, d_model)
        self.thinking_norm = nn.LayerNorm(d_model)
    
    def forward(self, q, k, v, mask=None, thinking_step=0):
        batch_size, seq_len = q.size(0), q.size(1)
        
        # 选择当前思考步骤的权重
        think_idx = min(thinking_step, len(self.w_q) - 1)
        
        Q = self.w_q[think_idx](q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k[think_idx](k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v[think_idx](v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        
        # 思考门控：结合原始输入和注意力输出
        if thinking_step > 0:
            gate_input = torch.cat([q, output], dim=-1)
            gate = torch.sigmoid(self.thinking_gate(gate_input))
            output = gate * output + (1 - gate) * q
            output = self.thinking_norm(output)
        
        return output, attn_weights

class ReasoningBlock(nn.Module):
    """推理块，支持多轮思考"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(ReasoningBlock, self).__init__()
        self.attention = DepthThinkingAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 多轮推理的前馈网络
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),  # 使用GELU激活函数
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            ) for _ in range(2)  # 两个推理步骤
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.thinking_steps = 2  # 每个块的思考步骤
    
    def forward(self, x, mask=None, thinking_depth=0):
        # 多轮思考
        current_x = x
        for step in range(self.thinking_steps):
            attn_output, _ = self.attention(current_x, current_x, current_x, mask, step)
            current_x = self.norm1(current_x + self.dropout(attn_output))
            
            ff_output = self.ffn[min(step, len(self.ffn)-1)](current_x)
            current_x = self.norm2(current_x + self.dropout(ff_output))
        
        return current_x

class EnhancedWikipediaAI(nn.Module):
    """增强版维基百科AI，支持深度思考"""
    
    def __init__(self, config):
        super(EnhancedWikipediaAI, self).__init__()
        self.config = config
        
        # 词嵌入和位置编码
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # 知识感知嵌入（用于整合外部知识）
        self.knowledge_embedding = nn.Linear(config.d_model * 2, config.d_model)
        
        # 推理层
        self.reasoning_layers = nn.ModuleList([
            ReasoningBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        
        # 置信度预测
        self.confidence_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.register_buffer("position_ids", 
                           torch.arange(config.max_seq_len).unsqueeze(0))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def integrate_knowledge(self, token_embeds, knowledge_vectors):
        """整合外部知识"""
        if knowledge_vectors is not None and len(knowledge_vectors) > 0:
            # 平均知识向量
            knowledge_embed = torch.mean(knowledge_vectors, dim=0).unsqueeze(0).unsqueeze(0)
            knowledge_embed = knowledge_embed.expand(token_embeds.size(0), token_embeds.size(1), -1)
            
            # 结合token嵌入和知识嵌入
            combined = torch.cat([token_embeds, knowledge_embed], dim=-1)
            return self.knowledge_embedding(combined)
        return token_embeds
    
    def forward(self, input_ids, attention_mask=None, knowledge_vectors=None, thinking_depth=0):
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入 + 位置嵌入
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(self.position_ids[:, :seq_len])
        x = token_embeds + position_embeds
        
        # 整合外部知识
        if self.config.knowledge_integration:
            x = self.integrate_knowledge(x, knowledge_vectors)
        
        # 创建注意力掩码
        if attention_mask is None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len))
            if input_ids.is_cuda:
                causal_mask = causal_mask.cuda()
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        
        # 多轮推理
        for i, layer in enumerate(self.reasoning_layers):
            layer_thinking_depth = min(thinking_depth, self.config.thinking_depth)
            x = layer(x, attention_mask, layer_thinking_depth)
        
        # 输出
        x = self.output_norm(x)
        logits = self.output_layer(x)
        
        # 预测置信度
        confidence = self.confidence_predictor(x.mean(dim=1))
        
        return logits, confidence
    
    def generate_with_thinking(self, input_ids, knowledge_vectors=None, max_length=100, 
                              temperature=0.8, top_k=50, min_confidence=0.5):
        """带深度思考的文本生成"""
        self.eval()
        
        with torch.no_grad():
            generated = input_ids.clone()
            thinking_log = []
            
            for step in range(max_length):
                if generated.size(1) > self.config.max_seq_len:
                    model_input = generated[:, -self.config.max_seq_len:]
                else:
                    model_input = generated
                
                # 多轮思考生成
                best_token = None
                best_confidence = 0
                
                for thinking_depth in range(self.config.thinking_depth + 1):
                    logits, confidence = self.forward(
                        model_input, knowledge_vectors=knowledge_vectors, 
                        thinking_depth=thinking_depth
                    )
                    
                    current_confidence = confidence.item()
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    if top_k is not None:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # 记录思考过程
                    thinking_log.append({
                        'step': step,
                        'thinking_depth': thinking_depth,
                        'confidence': current_confidence,
                        'token': next_token.item()
                    })
                    
                    # 选择置信度最高的结果
                    if current_confidence > best_confidence:
                        best_confidence = current_confidence
                        best_token = next_token
                
                # 如果置信度太低，提前结束
                if best_confidence < min_confidence and step > 10:
                    break
                
                # 添加最佳token到序列
                generated = torch.cat([generated, best_token], dim=1)
                
                # 结束标记
                if (best_token == 2).all():
                    break
            
            return generated, thinking_log, best_confidence

# ==================== 知识检索系统 ====================

class KnowledgeRetrievalSystem:
    """知识检索系统，用于增强AI的知识理解"""
    
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.entity_cache = {}
    
    def retrieve_related_knowledge(self, query, max_relations=10):
        """检索与查询相关的知识"""
        related_knowledge = []
        query_entities = self.extract_entities(query)
        
        for entity in query_entities:
            if entity in self.entity_cache:
                related_knowledge.extend(self.entity_cache[entity])
            else:
                # 在知识图谱中查找相关关系
                entity_relations = []
                for relation, objects in self.knowledge_graph.items():
                    if entity in relation:
                        for obj in objects[:3]:  # 取前3个相关对象
                            entity_relations.append({
                                'relation': relation,
                                'object': obj['object'],
                                'confidence': obj['confidence']
                            })
                
                # 缓存结果
                self.entity_cache[entity] = entity_relations
                related_knowledge.extend(entity_relations)
        
        # 按置信度排序并去重
        unique_relations = {}
        for rel in related_knowledge:
            key = f"{rel['relation']}_{rel['object']}"
            if key not in unique_relations or rel['confidence'] > unique_relations[key]['confidence']:
                unique_relations[key] = rel
        
        sorted_relations = sorted(unique_relations.values(), 
                                 key=lambda x: x['confidence'], reverse=True)
        
        return sorted_relations[:max_relations]
    
    def extract_entities(self, text):
        """从文本中提取实体"""
        # 简单的实体提取（实际可以使用NER模型）
        entities = re.findall(r'[\u4e00-\u9fa5]{2,6}', text)
        return [e for e in entities if len(e) >= 2]
    
    def create_knowledge_vectors(self, knowledge_relations, embedding_dim):
        """创建知识向量"""
        if not knowledge_relations:
            return None
        
        # 简单的知识向量生成（实际可以使用预训练的词向量）
        vectors = []
        for rel in knowledge_relations:
            # 基于关系文本生成简单向量
            relation_text = rel['relation'] + " " + rel['object']
            vector = self.text_to_vector(relation_text, embedding_dim)
            vectors.append(vector * rel['confidence'])  # 用置信度加权
        
        return torch.stack(vectors) if vectors else None
    
    def text_to_vector(self, text, dim):
        """将文本转换为向量（简化版）"""
        # 创建简单的哈希向量
        vector = np.zeros(dim)
        for i, char in enumerate(text):
            if i >= dim:
                break
            hash_val = hash(char) % 100 / 100.0
            vector[i] = hash_val
        return torch.tensor(vector, dtype=torch.float)

# ==================== 持续学习系统 ====================

class ContinuousLearningSystem:
    """持续学习系统，支持增量学习和知识更新"""
    
    def __init__(self):
        self.learning_log = []
        self.performance_history = []
        self.knowledge_gaps = []
    
    def log_interaction(self, query, response, confidence, feedback=None):
        """记录交互日志"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'confidence': confidence,
            'feedback': feedback,
            'hash': hashlib.md5(query.encode()).hexdigest()
        }
        self.learning_log.append(interaction)
        
        # 如果置信度低，记录知识缺口
        if confidence < 0.5:
            self.knowledge_gaps.append({
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence
            })
    
    def analyze_knowledge_gaps(self):
        """分析知识缺口"""
        gap_analysis = {}
        for gap in self.knowledge_gaps[-100:]:  # 分析最近100个缺口
            entities = re.findall(r'[\u4e00-\u9fa5]{2,6}', gap['query'])
            for entity in entities:
                if entity not in gap_analysis:
                    gap_analysis[entity] = []
                gap_analysis[entity].append(gap['confidence'])
        
        # 计算平均置信度
        gap_scores = {}
        for entity, confidences in gap_analysis.items():
            gap_scores[entity] = sum(confidences) / len(confidences)
        
        return sorted(gap_scores.items(), key=lambda x: x[1])[:10]  # 返回最需要学习的10个实体
    
    def save_learning_data(self):
        """保存学习数据"""
        with open(config.learning_log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'learning_log': self.learning_log[-1000:],  # 保存最近1000条
                'knowledge_gaps': self.knowledge_gaps[-500:],  # 保存最近500个缺口
                'performance_history': self.performance_history
            }, f, ensure_ascii=False, indent=2)
    
    def load_learning_data(self):
        """加载学习数据"""
        try:
            with open(config.learning_log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.learning_log = data.get('learning_log', [])
                self.knowledge_gaps = data.get('knowledge_gaps', [])
                self.performance_history = data.get('performance_history', [])
            print(f"学习数据已加载: {len(self.learning_log)} 条记录")
        except FileNotFoundError:
            print("学习数据文件不存在")

# ==================== 增强的训练系统 ====================

class EnhancedAITrainingSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.crawler = EnhancedWikipediaCrawler()
        self.tokenizer = Tokenizer()
        self.model = None
        self.knowledge_system = None
        self.learning_system = ContinuousLearningSystem()
        self.dataloader = None
        
    def prepare_training_data(self):
        """准备训练数据"""
        print("\n=== 准备训练数据 ===")
        
        if not self.crawler.collected_data:
            print("没有数据可用，请先运行爬虫")
            return False
        
        # 构建词汇表
        self.tokenizer.build_vocab(self.crawler.collected_data, config.vocab_size)
        config.vocab_size = len(self.tokenizer.vocab)
        print(f"词汇表大小: {config.vocab_size}")
        
        # 创建数据集
        dataset = WikiDataset(self.crawler.collected_data, self.tokenizer, config.max_seq_len)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        print(f"训练数据: {len(dataset)} 个样本")
        return True
    
    def crawl_with_knowledge_graph(self, max_topics=None):
        """爬取数据并构建知识图谱"""
        print("=== 开始增强爬取 ===")
        
        topics_to_crawl = self.crawler.topics[:max_topics] if max_topics else self.crawler.topics
        
        for i, topic in enumerate(topics_to_crawl, 1):
            print(f"\n[{i}/{len(topics_to_crawl)}] ", end="")
            self.crawler.crawl_topic(topic)
        
        # 保存数据和知识图谱
        self.crawler.save_data()
        self.crawler.save_knowledge_graph()
        
        print(f"\n爬取完成! 共收集 {len(self.crawler.collected_data)} 个页面")
        return len(self.crawler.collected_data)
    
    def train_enhanced_model(self):
        """训练增强版模型"""
        print("\n=== 开始增强训练 ===")
        
        # 准备数据
        if not self.prepare_training_data():
            return False
        
        # 初始化知识检索系统
        self.knowledge_system = KnowledgeRetrievalSystem(self.crawler.knowledge_graph)
        
        # 初始化模型
        self.model = EnhancedWikipediaAI(config).to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"增强模型参数: {total_params:,}")
        
        # 优化器和学习率调度
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
        
        # 训练循环
        self.model.train()
        for epoch in range(config.epochs):
            total_loss = 0
            total_confidence = 0
            start_time = time.time()
            
            for batch_idx, batch in enumerate(self.dataloader):
                batch = batch.to(self.device)
                
                # 准备输入和目标
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                # 为每个样本检索相关知识
                knowledge_vectors = []
                for i in range(inputs.size(0)):
                    # 解码输入文本用于知识检索
                    input_text = self.tokenizer.decode(inputs[i].cpu().tolist())
                    relations = self.knowledge_system.retrieve_related_knowledge(input_text)
                    vectors = self.knowledge_system.create_knowledge_vectors(relations, config.d_model)
                    knowledge_vectors.append(vectors)
                
                # 前向传播（带深度思考）
                optimizer.zero_grad()
                outputs, confidence = self.model(inputs, knowledge_vectors=knowledge_vectors, 
                                               thinking_depth=random.randint(0, config.thinking_depth))
                
                loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_confidence += confidence.mean().item()
                
                if batch_idx % 10 == 0:
                    avg_conf = total_confidence / (batch_idx + 1)
                    print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Conf: {avg_conf:.4f}')
            
            # 更新学习率
            scheduler.step()
            
            avg_loss = total_loss / len(self.dataloader)
            avg_confidence = total_confidence / len(self.dataloader)
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}, 平均置信度: {avg_confidence:.4f}, 时间: {epoch_time:.2f}s')
            
            # 记录性能
            self.learning_system.performance_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'confidence': avg_confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            # 测试深度思考生成
            self.test_thinking_generation(epoch + 1)
        
        return True
    
    def test_thinking_generation(self, epoch):
        """测试深度思考生成"""
        if self.model is None:
            return
        
        print(f"\nEpoch {epoch} 深度思考测试:")
        test_prompts = ["人工智能的未来发展", "机器学习的应用领域", "深度学习的原理"]
        
        for prompt in test_prompts:
            print(f"\n思考过程: '{prompt}'")
            
            # 检索相关知识
            relations = self.knowledge_system.retrieve_related_knowledge(prompt)
            knowledge_vectors = self.knowledge_system.create_knowledge_vectors(relations, config.d_model)
            
            tokens = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                generated, thinking_log, confidence = self.model.generate_with_thinking(
                    input_tensor, knowledge_vectors, max_length=80, 
                    temperature=0.7, min_confidence=config.confidence_threshold
                )
                
                text = self.tokenizer.decode(generated[0].cpu().tolist())
                print(f"  最终回答 (置信度: {confidence:.4f}): '{text}'")
                
                # 显示思考过程
                if thinking_log and len(thinking_log) > 0:
                    print("  思考日志:")
                    for i, log in enumerate(thinking_log[-5:]):  # 显示最后5步思考
                        token_str = self.tokenizer.decode([log['token']])
                        print(f"    步骤{log['step']}-深度{log['thinking_depth']}: {token_str} (置信度: {log['confidence']:.4f})")
    
    def save_model(self):
        """保存模型"""
        if self.model is None:
            print("没有模型可保存")
            return
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'config': config,
            'tokenizer_vocab': self.tokenizer.vocab,
            'training_info': f"基于 {len(self.crawler.collected_data)} 个维基百科页面训练"
        }
        
        torch.save(model_data, config.model_file)
        print(f"模型已保存到: {config.model_file}")
    
    def load_model(self):
        """加载模型"""
        try:
            checkpoint = torch.load(config.model_file, map_location=self.device)
            
            self.model = EnhancedWikipediaAI(checkpoint['config']).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.tokenizer.vocab = checkpoint['tokenizer_vocab']
            self.tokenizer.id_to_token = {v: k for k, v in self.tokenizer.vocab.items()}
            
            print(f"模型加载成功: {checkpoint['training_info']}")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

# ==================== 智能对话系统 ====================

class IntelligentDialogueSystem:
    """智能对话系统，集成所有增强功能"""
    
    def __init__(self, training_system):
        self.training_system = training_system
        self.conversation_history = []
        self.dialogue_context = ""
        
    def chat(self, user_input, thinking_mode=True):
        """智能对话"""
        if self.training_system.model is None:
            if not self.training_system.load_model():
                return "请先训练或加载模型", 0.0
        
        print(f"用户: {user_input}")
        
        # 更新对话上下文
        self.dialogue_context += f"用户: {user_input}\n"
        
        # 检索相关知识
        relations = self.training_system.knowledge_system.retrieve_related_knowledge(user_input)
        knowledge_vectors = self.training_system.knowledge_system.create_knowledge_vectors(
            relations, config.d_model
        )
        
        # 准备输入
        context_tokens = self.training_system.tokenizer.encode(self.dialogue_context[-500:])  # 限制上下文长度
        input_tensor = torch.tensor([context_tokens], dtype=torch.long, 
                                  device=self.training_system.device)
        
        # 生成回复
        with torch.no_grad():
            if thinking_mode:
                generated, thinking_log, confidence = self.training_system.model.generate_with_thinking(
                    input_tensor, knowledge_vectors, max_length=150, 
                    temperature=0.7, min_confidence=config.confidence_threshold
                )
                
                # 记录思考过程
                if thinking_log:
                    print("思考过程:")
                    for log in thinking_log[-3:]:  # 显示最后3步思考
                        token_str = self.training_system.tokenizer.decode([log['token']])
                        print(f"  深度{log['thinking_depth']}: {token_str}")
            else:
                # 使用普通生成方法
                logits, confidence = self.training_system.model(input_tensor, knowledge_vectors=knowledge_vectors)
                next_token_logits = logits[:, -1, :] / 0.7
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([input_tensor, next_token], dim=1)
                confidence = confidence.item()
        
        response = self.training_system.tokenizer.decode(generated[0].cpu().tolist())
        response = response[len(self.dialogue_context):].split('\n')[0]  # 提取新生成的部分
        
        # 更新对话历史
        self.dialogue_context += f"AI: {response}\n"
        self.conversation_history.append({
            'user': user_input,
            'ai': response,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        # 记录学习数据
        self.training_system.learning_system.log_interaction(user_input, response, confidence)
        
        return response, confidence

# ==================== 主程序 ====================

def main():
    system = EnhancedAITrainingSystem()
    dialogue_system = IntelligentDialogueSystem(system)
    
    print("=" * 60)
    print("       增强版维基百科AI系统 - 深度思考与持续学习")
    print("=" * 60)
    
    # 加载现有数据
    system.crawler.load_data()
    system.crawler.load_knowledge_graph()
    system.learning_system.load_learning_data()
    
    while True:
        print("\n请选择操作:")
        print("1. 增强爬取（构建知识图谱）")
        print("2. 增强训练（深度思考）")
        print("3. 智能对话")
        print("4. 分析知识缺口")
        print("5. 系统状态")
        print("6. 退出")
        
        choice = input("请输入选择 (1-6): ").strip()
        
        if choice == '1':
            max_topics = input("输入要爬取的主题数量 (默认10): ").strip()
            max_topics = int(max_topics) if max_topics.isdigit() else 10
            system.crawl_with_knowledge_graph(max_topics)
        
        elif choice == '2':
            if system.prepare_training_data():
                system.train_enhanced_model()
                system.save_model()
        
        elif choice == '3':
            print("\n=== 智能对话模式 ===")
            print("输入 '退出' 结束对话")
            print("输入 '思考模式' 切换深度思考")
            print("-" * 40)
            
            thinking_mode = True
            while True:
                user_input = input("\n你: ").strip()
                
                if user_input.lower() in ['退出', 'exit', 'quit']:
                    break
                elif user_input.lower() == '思考模式':
                    thinking_mode = not thinking_mode
                    mode = "开启" if thinking_mode else "关闭"
                    print(f"深度思考模式已{mode}")
                    continue
                
                response, confidence = dialogue_system.chat(user_input, thinking_mode)
                print(f"AI (置信度: {confidence:.4f}): {response}")
        
        elif choice == '4':
            gaps = system.learning_system.analyze_knowledge_gaps()
            if gaps:
                print("\n知识缺口分析 (需要重点学习的领域):")
                for entity, score in gaps:
                    print(f"  {entity}: 平均置信度 {score:.4f}")
            else:
                print("暂无显著知识缺口")
        
        elif choice == '5':
            print(f"\n系统状态:")
            print(f"  已收集数据: {len(system.crawler.collected_data)} 条")
            print(f"  知识图谱: {len(system.crawler.knowledge_graph)} 个关系")
            print(f"  学习记录: {len(system.learning_system.learning_log)} 条")
            print(f"  知识缺口: {len(system.learning_system.knowledge_gaps)} 个")
        
        elif choice == '6':
            # 保存学习数据
            system.learning_system.save_learning_data()
            print("学习数据已保存，再见！")
            break
        
        else:
            print("无效选择")

if __name__ == "__main__":
    main()