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

# ==================== SYSTEM INFORMATION ====================
SYSTEM_NAME = "WiseAbyss AI"
VERSION = "2.0"
MOTTO = "Deep Thinking, Boundless Knowledge"

def display_welcome():
    print("=" * 60)
    print(f"            {SYSTEM_NAME} v{VERSION}")
    print(f"            {MOTTO}")
    print("=" * 60)

# ==================== ENHANCED CONFIGURATION ====================
class EnhancedConfig:
    def __init__(self):
        # Model configuration
        self.vocab_size = 15000
        self.d_model = 384
        self.n_heads = 6
        self.n_layers = 6
        self.d_ff = 1024
        self.max_seq_len = 512
        self.dropout = 0.1
        self.batch_size = 2
        self.gradient_accumulation = 8
        
        # Training configuration
        self.epochs = 8
        self.learning_rate = 5e-5
        self.warmup_steps = 1000
        
        # Inference configuration
        self.thinking_depth = 3
        self.knowledge_integration = True
        self.confidence_threshold = 0.7
        
        # Crawler configuration
        self.max_pages_per_topic = 15
        self.request_delay = 1
        self.timeout = 10
        self.retry_count = 3
        
        # Knowledge graph configuration
        self.min_entity_freq = 2
        self.max_relations = 10000
        
        # File paths
        self.data_file = "enhanced_wiki_data.json"
        self.model_file = "enhanced_ai_model.pth"
        self.knowledge_graph_file = "knowledge_graph.json"
        self.learning_log_file = "learning_log.json"

config = EnhancedConfig()

# ==================== TOKENIZER CLASS ====================

class Tokenizer:
    """Tokenizer for text encoding and decoding"""
    
    def __init__(self):
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, texts, max_vocab_size=15000):
        """Build vocabulary from texts"""
        char_counter = Counter()
        
        for item in texts:
            if isinstance(item, dict):
                text = item['content']
            else:
                text = item
            char_counter.update(text)
        
        # Add most common characters
        for char, count in char_counter.most_common(max_vocab_size - 4):
            self.vocab[char] = len(self.vocab)
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        return self.vocab
    
    def encode(self, text):
        """Encode text to token IDs"""
        return [self.vocab.get(char, 1) for char in text]  # 1 is <UNK>
    
    def decode(self, tokens):
        """Decode token IDs to text"""
        return ''.join([self.id_to_token.get(token, '<UNK>') for token in tokens 
                       if token not in [0, 1, 2, 3]])

# ==================== DATASET CLASS ====================

class WikiDataset(Dataset):
    """Wikipedia dataset for training"""
    
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

# ==================== ENHANCED WIKIPEDIA CRAWLER ====================

class EnhancedWikipediaCrawler:
    """Enhanced Wikipedia crawler with knowledge graph construction"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (WiseAbyss-AI/2.0)'
        })
        self.collected_data = []
        self.knowledge_graph = defaultdict(list)
        self.entity_freq = Counter()
        
        # Knowledge domains for crawling
        self.topics = [
            # Computer Science
            "Artificial Intelligence", "Machine Learning", "Deep Learning", "Neural Networks", 
            "Natural Language Processing", "Computer Vision", "Data Science", "Big Data",
            "Cloud Computing", "Blockchain", "Algorithms", "Data Structures",
            "Programming Languages", "Software Engineering", "Operating Systems",
            
            # Mathematics & Science
            "Mathematics", "Physics", "Chemistry", "Biology", "Astronomy",
            "Statistics", "Probability Theory", "Calculus", "Linear Algebra", "Quantum Mechanics",
            
            # Humanities & Social Sciences
            "History", "Philosophy", "Psychology", "Economics", "Sociology",
            "Literature", "Art", "Music", "Political Science", "Law",
            
            # Technology & Engineering
            "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", 
            "Aerospace Engineering", "Robotics", "Internet of Things", "5G Technology",
            "Nanotechnology", "Biotechnology", "Renewable Energy"
        ]
    
    def extract_entities(self, text, title):
        """Extract entities and relationships from text"""
        entities = []
        relations = []
        
        # Simple entity extraction (can be replaced with NER models)
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if len(sentence) < 20:  # Increased for English
                continue
                
            # Extract potential entities (words with proper length)
            words = re.findall(r'\b[A-Za-z]{3,15}\b', sentence)
            for word in words:
                if len(word) >= 3:
                    self.entity_freq[word] += 1
                    entities.append(word)
            
            # Extract simple relationships (subject, relation, object)
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
        """Get page content and extract knowledge"""
        try:
            content_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'prop': 'extracts|links',
                'titles': title,
                'explaintext': True,
                'format': 'json',
                'pllimit': 20  # Get related links
            }
            
            response = self.session.get(content_url, params=params, timeout=config.timeout)
            if response.status_code == 200:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})
                
                for page_id, page_data in pages.items():
                    if 'extract' in page_data:
                        content = self.clean_content(page_data['extract'])
                        
                        # Extract entities and relationships
                        entities, relations = self.extract_entities(content, title)
                        
                        # Update knowledge graph
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
            print(f"Failed to get page '{title}' content: {e}")
        return None
    
    def clean_content(self, content):
        """Enhanced content cleaning"""
        # Remove reference markers
        content = re.sub(r'\[\d+\]', '', content)
        # Remove templates
        content = re.sub(r'\{\{.*?\}\}', '', content)
        # Remove file links
        content = re.sub(r'\[\[File:.*?\]\]', '', content)
        # Handle internal links [[link|display text]]
        content = re.sub(r'\[\[.*?\|(.*?)\]\]', r'\1', content)
        content = re.sub(r'\[\[(.*?)\]\]', r'\1', content)
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        return content.strip()
    
    def crawl_topic(self, topic):
        """Crawl specific topic and build knowledge graph"""
        print(f"Crawling topic: {topic}")
        
        page_titles = self.get_page_links(topic)
        if not page_titles:
            print(f"  No pages found for '{topic}'")
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
                print(f"  ✓ Crawled: {title} (Entities: {len(result['entities'])}, Relations: {len(result['relations'])})")
            
            time.sleep(config.request_delay)
    
    def get_page_links(self, topic):
        """Get related page links"""
        for attempt in range(config.retry_count):
            try:
                search_url = "https://en.wikipedia.org/w/api.php"
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
                print(f"Search for topic '{topic}' failed (attempt {attempt+1}): {e}")
                time.sleep(2)
        return []
    
    def save_data(self, filename=None):
        """Save collected data"""
        if filename is None:
            filename = config.data_file
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.collected_data, f, ensure_ascii=False, indent=2)
        print(f"Data saved to: {filename}")
    
    def load_data(self, filename=None):
        """Load collected data"""
        if filename is None:
            filename = config.data_file
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.collected_data = json.load(f)
            print(f"Loaded {len(self.collected_data)} records from {filename}")
        except FileNotFoundError:
            print(f"Data file {filename} does not exist")
            self.collected_data = []
    
    def save_knowledge_graph(self):
        """Save knowledge graph"""
        # Limit number of relations
        sorted_relations = sorted(self.knowledge_graph.items(), 
                                 key=lambda x: len(x[1]), reverse=True)
        limited_graph = dict(sorted_relations[:config.max_relations])
        
        with open(config.knowledge_graph_file, 'w', encoding='utf-8') as f:
            json.dump(limited_graph, f, ensure_ascii=False, indent=2)
        print(f"Knowledge graph saved: {config.knowledge_graph_file} (Relations: {len(limited_graph)})")
    
    def load_knowledge_graph(self):
        """Load knowledge graph"""
        try:
            with open(config.knowledge_graph_file, 'r', encoding='utf-8') as f:
                self.knowledge_graph = json.load(f)
            print(f"Knowledge graph loaded: {len(self.knowledge_graph)} relations")
        except FileNotFoundError:
            print("Knowledge graph file does not exist")

# ==================== DEEP THINKING AI MODEL ====================

class DepthThinkingAttention(nn.Module):
    """Deep thinking attention mechanism"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(DepthThinkingAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Multiple attention weights for multi-step thinking
        self.w_q = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.w_k = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.w_v = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Thinking gate
        self.thinking_gate = nn.Linear(d_model * 2, d_model)
        self.thinking_norm = nn.LayerNorm(d_model)
    
    def forward(self, q, k, v, mask=None, thinking_step=0):
        batch_size, seq_len = q.size(0), q.size(1)
        
        # Select current thinking step weights
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
        
        # Thinking gate: combine original input and attention output
        if thinking_step > 0:
            gate_input = torch.cat([q, output], dim=-1)
            gate = torch.sigmoid(self.thinking_gate(gate_input))
            output = gate * output + (1 - gate) * q
            output = self.thinking_norm(output)
        
        return output, attn_weights

class ReasoningBlock(nn.Module):
    """Reasoning block with multi-step thinking"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(ReasoningBlock, self).__init__()
        self.attention = DepthThinkingAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-step reasoning feed-forward networks
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),  # Using GELU activation
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            ) for _ in range(2)  # Two reasoning steps
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.thinking_steps = 2  # Thinking steps per block
    
    def forward(self, x, mask=None, thinking_depth=0):
        # Multi-step thinking
        current_x = x
        for step in range(self.thinking_steps):
            attn_output, _ = self.attention(current_x, current_x, current_x, mask, step)
            current_x = self.norm1(current_x + self.dropout(attn_output))
            
            ff_output = self.ffn[min(step, len(self.ffn)-1)](current_x)
            current_x = self.norm2(current_x + self.dropout(ff_output))
        
        return current_x

class EnhancedWikipediaAI(nn.Module):
    """Enhanced Wikipedia AI with deep thinking capabilities"""
    
    def __init__(self, config):
        super(EnhancedWikipediaAI, self).__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Knowledge-aware embedding (for integrating external knowledge)
        self.knowledge_embedding = nn.Linear(config.d_model * 2, config.d_model)
        
        # Reasoning layers
        self.reasoning_layers = nn.ModuleList([
            ReasoningBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Output layer
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        
        # Confidence predictor
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
        """Integrate external knowledge"""
        if knowledge_vectors is not None and len(knowledge_vectors) > 0:
            # Average knowledge vectors
            knowledge_embed = torch.mean(knowledge_vectors, dim=0).unsqueeze(0).unsqueeze(0)
            knowledge_embed = knowledge_embed.expand(token_embeds.size(0), token_embeds.size(1), -1)
            
            # Combine token embeddings and knowledge embeddings
            combined = torch.cat([token_embeds, knowledge_embed], dim=-1)
            return self.knowledge_embedding(combined)
        return token_embeds
    
    def forward(self, input_ids, attention_mask=None, knowledge_vectors=None, thinking_depth=0):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings + position embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(self.position_ids[:, :seq_len])
        x = token_embeds + position_embeds
        
        # Integrate external knowledge
        if self.config.knowledge_integration:
            x = self.integrate_knowledge(x, knowledge_vectors)
        
        # Create attention mask
        if attention_mask is None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len))
            if input_ids.is_cuda:
                causal_mask = causal_mask.cuda()
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        
        # Multi-step reasoning
        for i, layer in enumerate(self.reasoning_layers):
            layer_thinking_depth = min(thinking_depth, self.config.thinking_depth)
            x = layer(x, attention_mask, layer_thinking_depth)
        
        # Output
        x = self.output_norm(x)
        logits = self.output_layer(x)
        
        # Predict confidence
        confidence = self.confidence_predictor(x.mean(dim=1))
        
        return logits, confidence
    
    def generate_with_thinking(self, input_ids, knowledge_vectors=None, max_length=100, 
                              temperature=0.8, top_k=50, min_confidence=0.5):
        """Text generation with deep thinking"""
        self.eval()
        
        with torch.no_grad():
            generated = input_ids.clone()
            thinking_log = []
            
            for step in range(max_length):
                if generated.size(1) > self.config.max_seq_len:
                    model_input = generated[:, -self.config.max_seq_len:]
                else:
                    model_input = generated
                
                # Multi-step thinking generation
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
                    
                    # Record thinking process
                    thinking_log.append({
                        'step': step,
                        'thinking_depth': thinking_depth,
                        'confidence': current_confidence,
                        'token': next_token.item()
                    })
                    
                    # Select result with highest confidence
                    if current_confidence > best_confidence:
                        best_confidence = current_confidence
                        best_token = next_token
                
                # Early stop if confidence is too low
                if best_confidence < min_confidence and step > 10:
                    break
                
                # Add best token to sequence
                generated = torch.cat([generated, best_token], dim=1)
                
                # End token
                if (best_token == 2).all():
                    break
            
            return generated, thinking_log, best_confidence

# ==================== KNOWLEDGE RETRIEVAL SYSTEM ====================

class KnowledgeRetrievalSystem:
    """Knowledge retrieval system for enhancing AI understanding"""
    
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.entity_cache = {}
    
    def retrieve_related_knowledge(self, query, max_relations=10):
        """Retrieve knowledge related to query"""
        related_knowledge = []
        query_entities = self.extract_entities(query)
        
        for entity in query_entities:
            if entity in self.entity_cache:
                related_knowledge.extend(self.entity_cache[entity])
            else:
                # Find related relations in knowledge graph
                entity_relations = []
                for relation, objects in self.knowledge_graph.items():
                    if entity in relation:
                        for obj in objects[:3]:  # Take top 3 related objects
                            entity_relations.append({
                                'relation': relation,
                                'object': obj['object'],
                                'confidence': obj['confidence']
                            })
                
                # Cache results
                self.entity_cache[entity] = entity_relations
                related_knowledge.extend(entity_relations)
        
        # Sort by confidence and remove duplicates
        unique_relations = {}
        for rel in related_knowledge:
            key = f"{rel['relation']}_{rel['object']}"
            if key not in unique_relations or rel['confidence'] > unique_relations[key]['confidence']:
                unique_relations[key] = rel
        
        sorted_relations = sorted(unique_relations.values(), 
                                 key=lambda x: x['confidence'], reverse=True)
        
        return sorted_relations[:max_relations]
    
    def extract_entities(self, text):
        """Extract entities from text"""
        # Simple entity extraction (can use NER models)
        entities = re.findall(r'\b[A-Z][a-z]{2,}\b', text)  # Capitalized words
        return [e for e in entities if len(e) >= 3]
    
    def create_knowledge_vectors(self, knowledge_relations, embedding_dim):
        """Create knowledge vectors"""
        if not knowledge_relations:
            return None
        
        # Simple knowledge vector generation (can use pre-trained word vectors)
        vectors = []
        for rel in knowledge_relations:
            # Generate simple vector based on relation text
            relation_text = rel['relation'] + " " + rel['object']
            vector = self.text_to_vector(relation_text, embedding_dim)
            vectors.append(vector * rel['confidence'])  # Weight by confidence
        
        return torch.stack(vectors) if vectors else None
    
    def text_to_vector(self, text, dim):
        """Convert text to vector (simplified version)"""
        # Create simple hash-based vector
        vector = np.zeros(dim)
        for i, char in enumerate(text):
            if i >= dim:
                break
            hash_val = hash(char) % 100 / 100.0
            vector[i] = hash_val
        return torch.tensor(vector, dtype=torch.float)

# ==================== CONTINUOUS LEARNING SYSTEM ====================

class ContinuousLearningSystem:
    """Continuous learning system with incremental learning and knowledge updates"""
    
    def __init__(self):
        self.learning_log = []
        self.performance_history = []
        self.knowledge_gaps = []
    
    def log_interaction(self, query, response, confidence, feedback=None):
        """Log interaction"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'confidence': confidence,
            'feedback': feedback,
            'hash': hashlib.md5(query.encode()).hexdigest()
        }
        self.learning_log.append(interaction)
        
        # Record knowledge gap if confidence is low
        if confidence < 0.5:
            self.knowledge_gaps.append({
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence
            })
    
    def analyze_knowledge_gaps(self):
        """Analyze knowledge gaps"""
        gap_analysis = {}
        for gap in self.knowledge_gaps[-100:]:  # Analyze recent 100 gaps
            entities = re.findall(r'\b[A-Z][a-z]{2,}\b', gap['query'])
            for entity in entities:
                if entity not in gap_analysis:
                    gap_analysis[entity] = []
                gap_analysis[entity].append(gap['confidence'])
        
        # Calculate average confidence
        gap_scores = {}
        for entity, confidences in gap_analysis.items():
            gap_scores[entity] = sum(confidences) / len(confidences)
        
        return sorted(gap_scores.items(), key=lambda x: x[1])[:10]  # Top 10 entities to learn
    
    def save_learning_data(self):
        """Save learning data"""
        with open(config.learning_log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'learning_log': self.learning_log[-1000:],  # Save recent 1000 records
                'knowledge_gaps': self.knowledge_gaps[-500:],  # Save recent 500 gaps
                'performance_history': self.performance_history
            }, f, ensure_ascii=False, indent=2)
    
    def load_learning_data(self):
        """Load learning data"""
        try:
            with open(config.learning_log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.learning_log = data.get('learning_log', [])
                self.knowledge_gaps = data.get('knowledge_gaps', [])
                self.performance_history = data.get('performance_history', [])
            print(f"Learning data loaded: {len(self.learning_log)} records")
        except FileNotFoundError:
            print("Learning data file does not exist")

# ==================== ENHANCED TRAINING SYSTEM ====================

class EnhancedAITrainingSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.crawler = EnhancedWikipediaCrawler()
        self.tokenizer = Tokenizer()
        self.model = None
        self.knowledge_system = None
        self.learning_system = ContinuousLearningSystem()
        self.dataloader = None
        
    def prepare_training_data(self):
        """Prepare training data"""
        print("\n=== Preparing Training Data ===")
        
        if not self.crawler.collected_data:
            print("No data available, please run crawler first")
            return False
        
        # Build vocabulary
        self.tokenizer.build_vocab(self.crawler.collected_data, config.vocab_size)
        config.vocab_size = len(self.tokenizer.vocab)
        print(f"Vocabulary size: {config.vocab_size}")
        
        # Create dataset
        dataset = WikiDataset(self.crawler.collected_data, self.tokenizer, config.max_seq_len)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        print(f"Training data: {len(dataset)} samples")
        return True
    
    def crawl_with_knowledge_graph(self, max_topics=None):
        """Crawl data and build knowledge graph"""
        print("=== Starting Enhanced Crawling ===")
        
        topics_to_crawl = self.crawler.topics[:max_topics] if max_topics else self.crawler.topics
        
        for i, topic in enumerate(topics_to_crawl, 1):
            print(f"\n[{i}/{len(topics_to_crawl)}] ", end="")
            self.crawler.crawl_topic(topic)
        
        # Save data and knowledge graph
        self.crawler.save_data()
        self.crawler.save_knowledge_graph()
        
        print(f"\nCrawling completed! Collected {len(self.crawler.collected_data)} pages")
        return len(self.crawler.collected_data)
    
    def train_enhanced_model(self):
        """Train enhanced model"""
        print("\n=== Starting Enhanced Training ===")
        
        # Prepare data
        if not self.prepare_training_data():
            return False
        
        # Initialize knowledge retrieval system
        self.knowledge_system = KnowledgeRetrievalSystem(self.crawler.knowledge_graph)
        
        # Initialize model
        self.model = EnhancedWikipediaAI(config).to(self.device)
        
        # Print model information
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Enhanced model parameters: {total_params:,}")
        
        # Optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
        
        # Training loop
        self.model.train()
        for epoch in range(config.epochs):
            total_loss = 0
            total_confidence = 0
            start_time = time.time()
            
            for batch_idx, batch in enumerate(self.dataloader):
                batch = batch.to(self.device)
                
                # Prepare inputs and targets
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                # Retrieve related knowledge for each sample
                knowledge_vectors = []
                for i in range(inputs.size(0)):
                    # Decode input text for knowledge retrieval
                    input_text = self.tokenizer.decode(inputs[i].cpu().tolist())
                    relations = self.knowledge_system.retrieve_related_knowledge(input_text)
                    vectors = self.knowledge_system.create_knowledge_vectors(relations, config.d_model)
                    knowledge_vectors.append(vectors)
                
                # Forward pass (with deep thinking)
                optimizer.zero_grad()
                outputs, confidence = self.model(inputs, knowledge_vectors=knowledge_vectors, 
                                               thinking_depth=random.randint(0, config.thinking_depth))
                
                loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_confidence += confidence.mean().item()
                
                if batch_idx % 10 == 0:
                    avg_conf = total_confidence / (batch_idx + 1)
                    print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Conf: {avg_conf:.4f}')
            
            # Update learning rate
            scheduler.step()
            
            avg_loss = total_loss / len(self.dataloader)
            avg_confidence = total_confidence / len(self.dataloader)
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1} completed, Avg Loss: {avg_loss:.4f}, Avg Confidence: {avg_confidence:.4f}, Time: {epoch_time:.2f}s')
            
            # Record performance
            self.learning_system.performance_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'confidence': avg_confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            # Test deep thinking generation
            self.test_thinking_generation(epoch + 1)
        
        return True
    
    def test_thinking_generation(self, epoch):
        """Test deep thinking generation"""
        if self.model is None:
            return
        
        print(f"\nEpoch {epoch} Deep Thinking Test:")
        test_prompts = ["Future of artificial intelligence", "Applications of machine learning", "Principles of deep learning"]
        
        for prompt in test_prompts:
            print(f"\nThinking process: '{prompt}'")
            
            # Retrieve related knowledge
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
                print(f"  Final answer (Confidence: {confidence:.4f}): '{text}'")
                
                # Display thinking process
                if thinking_log and len(thinking_log) > 0:
                    print("  Thinking log:")
                    for i, log in enumerate(thinking_log[-5:]):  # Show last 5 thinking steps
                        token_str = self.tokenizer.decode([log['token']])
                        print(f"    Step{log['step']}-Depth{log['thinking_depth']}: {token_str} (Confidence: {log['confidence']:.4f})")
    
    def save_model(self):
        """Save model"""
        if self.model is None:
            print("No model to save")
            return
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'config': config,
            'tokenizer_vocab': self.tokenizer.vocab,
            'training_info': f"Trained on {len(self.crawler.collected_data)} Wikipedia pages"
        }
        
        torch.save(model_data, config.model_file)
        print(f"Model saved to: {config.model_file}")
    
    def load_model(self):
        """Load model"""
        try:
            checkpoint = torch.load(config.model_file, map_location=self.device)
            
            self.model = EnhancedWikipediaAI(checkpoint['config']).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.tokenizer.vocab = checkpoint['tokenizer_vocab']
            self.tokenizer.id_to_token = {v: k for k, v in self.tokenizer.vocab.items()}
            
            print(f"Model loaded successfully: {checkpoint['training_info']}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

# ==================== INTELLIGENT DIALOGUE SYSTEM ====================

class IntelligentDialogueSystem:
    """Intelligent dialogue system integrating all enhanced features"""
    
    def __init__(self, training_system):
        self.training_system = training_system
        self.conversation_history = []
        self.dialogue_context = ""
        
    def chat(self, user_input, thinking_mode=True):
        """Intelligent dialogue"""
        if self.training_system.model is None:
            if not self.training_system.load_model():
                return "Please train or load model first", 0.0
        
        print(f"User: {user_input}")
        
        # Update dialogue context
        self.dialogue_context += f"User: {user_input}\n"
        
        # Retrieve related knowledge
        relations = self.training_system.knowledge_system.retrieve_related_knowledge(user_input)
        knowledge_vectors = self.training_system.knowledge_system.create_knowledge_vectors(
            relations, config.d_model
        )
        
        # Prepare input
        context_tokens = self.training_system.tokenizer.encode(self.dialogue_context[-500:])  # Limit context length
        input_tensor = torch.tensor([context_tokens], dtype=torch.long, 
                                  device=self.training_system.device)
        
        # Generate response
        with torch.no_grad():
            if thinking_mode:
                generated, thinking_log, confidence = self.training_system.model.generate_with_thinking(
                    input_tensor, knowledge_vectors, max_length=150, 
                    temperature=0.7, min_confidence=config.confidence_threshold
                )
                
                # Record thinking process
                if thinking_log:
                    print("Thinking process:")
                    for log in thinking_log[-3:]:  # Show last 3 thinking steps
                        token_str = self.training_system.tokenizer.decode([log['token']])
                        print(f"  Depth{log['thinking_depth']}: {token_str}")
            else:
                # Use normal generation method
                logits, confidence = self.training_system.model(input_tensor, knowledge_vectors=knowledge_vectors)
                next_token_logits = logits[:, -1, :] / 0.7
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([input_tensor, next_token], dim=1)
                confidence = confidence.item()
        
        response = self.training_system.tokenizer.decode(generated[0].cpu().tolist())
        response = response[len(self.dialogue_context):].split('\n')[0]  # Extract newly generated part
        
        # Update conversation history
        self.dialogue_context += f"AI: {response}\n"
        self.conversation_history.append({
            'user': user_input,
            'ai': response,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        # Record learning data
        self.training_system.learning_system.log_interaction(user_input, response, confidence)
        
        return response, confidence

# ==================== MAIN PROGRAM ====================

def main():
    display_welcome()
    
    system = EnhancedAITrainingSystem()
    dialogue_system = IntelligentDialogueSystem(system)
    
    # Load existing data
    system.crawler.load_data()
    system.crawler.load_knowledge_graph()
    system.learning_system.load_learning_data()
    
    while True:
        print("\nPlease select an option:")
        print("1. Enhanced Crawling (Build Knowledge Graph)")
        print("2. Enhanced Training (Deep Thinking)")
        print("3. Intelligent Dialogue")
        print("4. Analyze Knowledge Gaps")
        print("5. System Status")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            max_topics = input("Enter number of topics to crawl (default 10): ").strip()
            max_topics = int(max_topics) if max_topics.isdigit() else 10
            system.crawl_with_knowledge_graph(max_topics)
        
        elif choice == '2':
            if system.prepare_training_data():
                system.train_enhanced_model()
                system.save_model()
        
        elif choice == '3':
            print("\n=== Intelligent Dialogue Mode ===")
            print("Enter 'exit' to end conversation")
            print("Enter 'thinking mode' to toggle deep thinking")
            print("-" * 40)
            
            thinking_mode = True
            while True:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                elif user_input.lower() == 'thinking mode':
                    thinking_mode = not thinking_mode
                    mode = "enabled" if thinking_mode else "disabled"
                    print(f"Deep thinking mode {mode}")
                    continue
                
                response, confidence = dialogue_system.chat(user_input, thinking_mode)
                print(f"AI (Confidence: {confidence:.4f}): {response}")
        
        elif choice == '4':
            gaps = system.learning_system.analyze_knowledge_gaps()
            if gaps:
                print("\nKnowledge Gap Analysis (Areas needing focus):")
                for entity, score in gaps:
                    print(f"  {entity}: Average confidence {score:.4f}")
            else:
                print("No significant knowledge gaps found")
        
        elif choice == '5':
            print(f"\nSystem Status:")
            print(f"  Collected data: {len(system.crawler.collected_data)} records")
            print(f"  Knowledge graph: {len(system.crawler.knowledge_graph)} relations")
            print(f"  Learning records: {len(system.learning_system.learning_log)}")
            print(f"  Knowledge gaps: {len(system.learning_system.knowledge_gaps)}")
        
        elif choice == '6':
            # Save learning data
            system.learning_system.save_learning_data()
            print("Learning data saved. Goodbye!")
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
