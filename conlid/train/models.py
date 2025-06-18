"""This file contains the definition of the following 4 models: LID-CE, LID-SCL, ConLID-S, ConLID-H"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import PyTorchModelHubMixin

# LID-CE definition
class LID_CE(nn.Module, PyTorchModelHubMixin):
    def __init__(self, vocab_size, embedding_size, num_classes, unk_id, pad_id, aggr, **kwargs):
        super(LID_CE, self).__init__()

        assert aggr in ['mean', 'max', 'sum']

        if aggr == 'mean':
            self.aggr_fn = lambda x, mask: torch.sum(x * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1, keepdim=True)
        elif aggr == 'max':
            self.aggr_fn = lambda x, mask: torch.max(x * mask.unsqueeze(-1) + (mask.unsqueeze(-1) - 1) * float('-inf'), dim=1)[0]
        else: # sum
            self.aggr_fn = lambda x, mask: torch.sum(x * mask.unsqueeze(-1), dim=1)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.unk_id = unk_id
        self.pad_id = pad_id
    
    def compute_embeddings_batch(self, ids_batch):
        """Compute embeddings for a batch of ids"""
        mask = ~((ids_batch == self.unk_id) | (ids_batch == self.pad_id))
        if not mask.any():
            raise Exception(f"NaN values in ids: {ids_batch}")
            
        embeddings = self.embedding(ids_batch)
        return self.aggr_fn(embeddings, mask)
        
    def forward(self, input_ids, labels=None):
        # input_ids
        embeddings = self.compute_embeddings_batch(input_ids) # batch_size x embed_size
        logits = self.fc(embeddings) # (batch_size, num_classes)

        if labels is None:
            # inference
            return logits

        ce_loss = self.criterion(logits, labels)
        
        return {
            "loss": ce_loss,
            "logits": logits
        }

# LID-SCL definition
class LID_SCL(nn.Module, PyTorchModelHubMixin):
    def __init__(self, vocab_size, embedding_size, num_classes, unk_id, pad_id, aggr, contrastive_temperature, **kwargs):
        super(LID_SCL, self).__init__()

        assert aggr in ['mean', 'max', 'sum']

        if aggr == 'mean':
            self.aggr_fn = lambda x, mask: torch.sum(x * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1, keepdim=True)
        elif aggr == 'max':
            self.aggr_fn = lambda x, mask: torch.max(x * mask.unsqueeze(-1) + (mask.unsqueeze(-1) - 1) * float('-inf'), dim=1)[0]
        else: # sum
            self.aggr_fn = lambda x, mask: torch.sum(x * mask.unsqueeze(-1), dim=1)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.contrastive_temperature = contrastive_temperature

        self.ce_loss_log = []
        self.contrastive_loss_log = []

        self.unk_id = unk_id
        self.pad_id = pad_id

    def reset_loss_log(self):
        self.ce_loss_log = []
        self.contrastive_loss_log = []

    def compute_embeddings_batch(self, ids_batch):
        """Compute embeddings for a batch of ids"""
        mask = ~((ids_batch == self.unk_id) | (ids_batch == self.pad_id))
        if not mask.any():
            raise Exception(f"NaN values in ids: {ids_batch}")
            
        embeddings = self.embedding(ids_batch)
        return self.aggr_fn(embeddings, mask)
        
    def compute_contrastive_loss(self, embeddings, labels):
        batch_size = embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0).to(embeddings.device)
        
        # Get all positive samples as:
        # (+) Other samples with the same label in the batch
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        pos_mask -= torch.diag(torch.ones((pos_mask.size()[0]))).to(pos_mask.device)
        
        if not (pos_mask == 1).any():
            return torch.tensor(0.0).to(embeddings.device)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.contrastive_temperature
        
        # Compute exp similarities directly
        exp_sim = torch.exp(sim_matrix)
        
        # Calculate positive pairs sum (numerator)
        pos_sim = exp_sim * pos_mask
        pos_sum = pos_sim.sum(dim=1)
        
        # Calculate all pairs sum (denominator)
        self_contrast_mask = 1 - torch.diag(torch.ones((pos_mask.size()[0]))).to(pos_mask.device)
        all_sum = exp_sim * self_contrast_mask
        all_sum = all_sum.sum(dim=1)
        
        # Calculate the average loss of examples with positive pairs only
        pos_idx = pos_sum != 0.0
        loss = -torch.log(pos_sum[pos_idx] / all_sum[pos_idx])
        return loss.mean()
    
    def forward(self, input_ids, labels=None):
        # input_ids
        embeddings = self.compute_embeddings_batch(input_ids) # batch_size x embed_size
        logits = self.fc(embeddings) # (batch_size, num_classes)

        if labels is None:
            # inference
            return logits

        ce_loss = self.criterion(logits, labels)
        contrastive_loss = self.compute_contrastive_loss(embeddings, labels)
        total_loss = ce_loss + contrastive_loss

        # Store losses for logging
        self.ce_loss_log.append(ce_loss.item())
        self.contrastive_loss_log.append(contrastive_loss.item())
        
        return {
            "loss": total_loss,
            "logits": logits
        }

class MemoryBank():
    def __init__(self, bank_size):
        self.bank_size = bank_size
        self.memory_embeddings = None
        self.memory_labels = None

    def dequeue(self):
        self.memory_embeddings = self.memory_embeddings[-self.bank_size:, :]
        self.memory_labels = self.memory_labels[-self.bank_size:]

    def enqueue(self, embeddings, labels):
        if self.memory_embeddings is None:
            self.memory_embeddings = embeddings.detach().clone()
            self.memory_labels = labels.clone()
        else:
            self.memory_embeddings = torch.cat([self.memory_embeddings, embeddings.detach()], dim=0)
            self.memory_labels = torch.cat([self.memory_labels, labels], dim=0)

        if self.memory_labels.size()[0] > self.bank_size:
            self.dequeue()

    def get_memory(self):
        return self.memory_embeddings, self.memory_labels

# ConLID-S definition
class ConLID_S(nn.Module, PyTorchModelHubMixin):
    def __init__(self, vocab_size, embedding_size, num_classes, unk_id, pad_id, aggr, contrastive_temperature, bank_size, **kwargs):
        super(ConLID_S, self).__init__()

        assert aggr in ['mean', 'max', 'sum']

        if aggr == 'mean':
            self.aggr_fn = lambda x, mask: torch.sum(x * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1, keepdim=True)
        elif aggr == 'max':
            self.aggr_fn = lambda x, mask: torch.max(x * mask.unsqueeze(-1) + (mask.unsqueeze(-1) - 1) * float('-inf'), dim=1)[0]
        else: # sum
            self.aggr_fn = lambda x, mask: torch.sum(x * mask.unsqueeze(-1), dim=1)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.contrastive_temperature = contrastive_temperature
        self.memory_bank = MemoryBank(bank_size = bank_size)

        self.ce_loss_log = []
        self.contrastive_loss_log = []
    
        self.unk_id = unk_id
        self.pad_id = pad_id

    def reset_loss_log(self):
        self.ce_loss_log = []
        self.contrastive_loss_log = []

    def compute_embeddings_batch(self, ids_batch):
        """Compute embeddings for a batch of ids"""
        mask = ~((ids_batch == self.unk_id) | (ids_batch == self.pad_id))
        if not mask.any():
            raise Exception(f"NaN values in ids: {ids_batch}")
            
        embeddings = self.embedding(ids_batch)
        return self.aggr_fn(embeddings, mask)
        
    def compute_contrastive_loss(self, embeddings, labels):
        # Enqueue / Dequeue the memory
        memory_embeddings, memory_labels = self.memory_bank.get_memory()
        if memory_embeddings is not None:
            self.memory_bank.enqueue(embeddings, labels)
            embeddings = torch.cat([embeddings, memory_embeddings], dim=0)
            labels = torch.cat([labels, memory_labels], dim=0)
        else:
            self.memory_bank.enqueue(embeddings, labels)

        batch_size = embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0).to(embeddings.device)
        
        # Get all positive samples as:
        # (+) Other samples with the same label in the batch
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        pos_mask -= torch.diag(torch.ones((pos_mask.size()[0]))).to(pos_mask.device)
        
        if not (pos_mask == 1).any():
            return torch.tensor(0.0).to(embeddings.device)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.contrastive_temperature
        
        # Compute exp similarities directly
        exp_sim = torch.exp(sim_matrix)
        
        # Calculate positive pairs sum (numerator)
        pos_sim = exp_sim * pos_mask
        pos_sum = pos_sim.sum(dim=1)
        
        # Calculate all pairs sum (denominator)
        self_contrast_mask = 1 - torch.diag(torch.ones((pos_mask.size()[0]))).to(pos_mask.device)
        all_sum = exp_sim * self_contrast_mask
        all_sum = all_sum.sum(dim=1)
        
        # Calculate the average loss of examples with positive pairs only
        pos_idx = pos_sum != 0.0
        loss = -torch.log(pos_sum[pos_idx] / all_sum[pos_idx])
        return loss.mean()
    
    def forward(self, input_ids, labels=None):
        # input_ids
        embeddings = self.compute_embeddings_batch(input_ids) # batch_size x embed_size
        logits = self.fc(embeddings) # (batch_size, num_classes)

        if labels is None:
            # inference
            return logits

        ce_loss = self.criterion(logits, labels)
        contrastive_loss = self.compute_contrastive_loss(embeddings, labels)
        total_loss = ce_loss + contrastive_loss

        # Store losses for logging
        self.ce_loss_log.append(ce_loss.item())
        self.contrastive_loss_log.append(contrastive_loss.item())
        
        return {
            "loss": total_loss,
            "logits": logits
        }

# ConLID-H definition
class ConLID_H(nn.Module, PyTorchModelHubMixin):
    def __init__(self, vocab_size, embedding_size, num_classes, unk_id, pad_id, aggr, contrastive_temperature, bank_size, min_neg):
        super(ConLID_H, self).__init__()

        assert aggr in ['mean', 'max', 'sum']

        if aggr == 'mean':
            self.aggr_fn = lambda x, mask: torch.sum(x * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1, keepdim=True)
        elif aggr == 'max':
            self.aggr_fn = lambda x, mask: torch.max(x * mask.unsqueeze(-1) + (mask.unsqueeze(-1) - 1) * float('-inf'), dim=1)[0]
        else: # sum
            self.aggr_fn = lambda x, mask: torch.sum(x * mask.unsqueeze(-1), dim=1)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.contrastive_temperature = contrastive_temperature
        self.memory_bank = MemoryBank(bank_size = bank_size)
        self.min_neg = min_neg

        self.ce_loss_log = []
        self.contrastive_loss_log = []
    
        self.unk_id = unk_id
        self.pad_id = pad_id
    
    def reset_loss_log(self):
        self.ce_loss_log = []
        self.contrastive_loss_log = []

    def compute_embeddings_batch(self, ids_batch):
        """Compute embeddings for a batch of ids"""
        mask = ~((ids_batch == self.unk_id) | (ids_batch == self.pad_id))
        if not mask.any():
            raise Exception(f"NaN values in ids: {ids_batch}")
            
        embeddings = self.embedding(ids_batch)
        return self.aggr_fn(embeddings, mask)

    def get_neg_mask(self, label_ids, doc_ids, dom_ids, dom_gen_ids, script_ids):
        """
        Generate a negative mask based on the specified conditions below:
            1. different label_id, same script_id, same doc_id
            2. different label_id, same script_id, same dom_ids
            3. different label_id, same script_id, same dom_gen_ids
            4. different label_id, same script_id
            5. different label_id, same doc_id
            6. different label_id, same dom_ids
            7. different label_id, same dom_gen_ids
            8. different label_id
        """
        device = label_ids.device
        batch_size = label_ids.size(0)
    
        # Initialize the mask
        neg_mask = torch.zeros((batch_size, batch_size), device=device)
    
        # Conditions for negatives
        neg_mask_label = (label_ids.unsqueeze(1) != label_ids.unsqueeze(0)).float()  # Different labels
        neg_mask_script = (script_ids.unsqueeze(1) == script_ids.unsqueeze(0)).float()  # Same script
        neg_mask_doc = (doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0)).float()  # Same doc
        neg_mask_dom = (dom_ids.unsqueeze(1) == dom_ids.unsqueeze(0)).float()  # Same dom
        neg_mask_dom_gen = (dom_gen_ids.unsqueeze(1) == dom_gen_ids.unsqueeze(0)).float()  # Same dom_gen
    
        # List of conditions in priority order
        conditions = [
            neg_mask_label * neg_mask_script * neg_mask_doc,  # 1
            neg_mask_label * neg_mask_script * neg_mask_dom,  # 2
            neg_mask_label * neg_mask_script * neg_mask_dom_gen,  # 3
            neg_mask_label * neg_mask_script, # 4
            neg_mask_label * neg_mask_doc,  # 5
            neg_mask_label * neg_mask_dom,  # 6
            neg_mask_label * neg_mask_dom_gen,  # 7
            neg_mask_label  # 8
        ]
    
        # Iterate over each row and apply conditions
        for i in range(batch_size):
            row_neg_mask = torch.zeros(batch_size, device=device)
            for condition in conditions:
                # Add negatives satisfying the current condition
                row_neg_mask = condition[i].clone()
    
                # Count the negatives added so far
                num_negatives = row_neg_mask.sum()
    
                # Stop if we have enough negatives for this row
                if num_negatives >= self.min_neg:
                    break
    
            # Ensure only valid negatives (binary mask)
            neg_mask[i] = (row_neg_mask > 0).float()
    
        return neg_mask
        
    def compute_contrastive_loss(self, embeddings, labels, doc_ids, dom_ids, dom_gen_ids, script_ids):
        # Enqueue / Dequeue the memory
        memory_embeddings, memory_labels, memory_doc_ids, memory_dom_ids, memory_dom_gen_ids, memory_script_ids = self.memory_bank.get_memory()

        if memory_embeddings is not None:
            self.memory_bank.enqueue(embeddings, labels, doc_ids, dom_ids, dom_gen_ids, script_ids)
            embeddings = torch.cat([embeddings, memory_embeddings], dim=0)
            labels = torch.cat([labels, memory_labels], dim=0)
            doc_ids = torch.cat([doc_ids, memory_doc_ids], dim=0)
            dom_ids = torch.cat([dom_ids, memory_dom_ids], dim=0)
            dom_gen_ids = torch.cat([dom_gen_ids, memory_dom_gen_ids], dim=0)
            script_ids = torch.cat([script_ids, memory_script_ids], dim=0)
        else:
            self.memory_bank.enqueue(embeddings, labels, doc_ids, dom_ids, dom_gen_ids, script_ids)

        batch_size = embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0).to(embeddings.device)
        
        # Get all positive pairs as:
        # (+) Other samples with the same label in the batch
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        pos_mask -= torch.diag(torch.ones((pos_mask.size()[0]))).to(pos_mask.device)

        # Get all negative pairs
        neg_mask = self.get_neg_mask(labels, doc_ids, dom_ids, dom_gen_ids, script_ids)
        
        if not (pos_mask == 1).any():
            return torch.tensor(0.0).to(embeddings.device)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.contrastive_temperature
        
        # Compute exp similarities directly
        exp_sim = torch.exp(sim_matrix)
        
        # Calculate numerator (positive pairs sum)
        pos_sim = exp_sim * pos_mask
        pos_sum = pos_sim.sum(dim=1)
        
        # Calculate denominator (positive and negative pairs sum)
        all_sum = exp_sim * (pos_mask + neg_mask)
        all_sum = all_sum.sum(dim=1)
        
        # Calculate the average loss of examples with positive pairs only
        pos_idx = pos_sum != 0.0
        loss = -torch.log(pos_sum[pos_idx] / all_sum[pos_idx])
        return loss.mean()
    
    def forward(self, input_ids, labels=None, doc_ids=None, dom_ids=None, dom_gen_ids=None, script_ids=None):
        # input_ids
        embeddings = self.compute_embeddings_batch(input_ids) # batch_size x embed_size
        logits = self.fc(embeddings) # (batch_size, num_classes)

        if labels is None:
            # inference
            return logits
        
        ce_loss = self.criterion(logits, labels)
        if script_ids is None:
            # evaluation -> No contrastive
            return {
                "loss": ce_loss,
                "logits": logits
            }

        contrastive_loss = self.compute_contrastive_loss(embeddings, labels, doc_ids, dom_ids, dom_gen_ids, script_ids)
        total_loss = ce_loss + contrastive_loss

        # Store losses for logging
        self.ce_loss_log.append(ce_loss.item())
        self.contrastive_loss_log.append(contrastive_loss.item())
        
        return {
            "loss": total_loss,
            "logits": logits
        }