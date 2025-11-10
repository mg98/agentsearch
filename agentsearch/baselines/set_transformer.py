import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from agentsearch.utils.globals import get_torch_device
import wandb

SetTransformerData = tuple[Agent, Question, float]

class SetEncoder(nn.Module):
    """Encodes an unordered set of (q_emb , score) pairs."""
    def __init__(self, question_embedding_dim, hidden_dim, n_heads, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(question_embedding_dim + 1, hidden_dim)  # (question,scores) -> embedding
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, dim_feedforward=128) # feedforward dim can be tuned
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, rows):  # rows: (B: batch, N-1, D: q_emb_dim + score_dim)
        x = self.input_proj(rows)                # (B,N-1,D)
        x = x.transpose(0,1)                     # transformer expects (N,B,D)
        encoded = self.encoder(x)                # (N-1,B,D)
        pooled = encoded.mean(dim=0)             # permutation-invariant pooling
        return pooled                            # (B,D)

class TabularPredictor(nn.Module):
    def __init__(self, question_embedding_dim, agent_embedding_dim, hidden_dim, n_heads, n_layers):
        super().__init__()
        self.agent_embedder = nn.Linear(agent_embedding_dim, hidden_dim)  # table-level b
        self.set_encoder = SetEncoder(
            question_embedding_dim=question_embedding_dim, 
            hidden_dim=hidden_dim, 
            n_heads=n_heads,
            n_layers=n_layers
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*2 + question_embedding_dim, hidden_dim),    # pooled set + agent + question N
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)                                         # predict score for question N
        )

    def forward(self, context_questions_with_scores, agent_emb, tgt_question):
        """
        We have N questions in total, the first N-1 are context, i.e. we know the score, the last one is the target, i.e. we want to predict the score.

        context_questions: (B, N-1, Q + 1) (B is batch size, N-1 is number of context questions, Q is question embedding size and 1 for the score)
        agent_emb: (B, A) the embedding of the target agent (the same for each question in the context)
        tgt_question: (B, Q) the embedding of the target question (for which we want to predict the score)
        """
        # context may be empty
        if context_questions_with_scores.shape[1] > 0:
            context_enc = self.set_encoder(context_questions_with_scores)        # (B,D) 
        else:
            context_enc = torch.zeros((agent_emb.shape[0], self.set_encoder.hidden_dim), device=agent_emb.device)  # (B,D)

        agent_enc = self.agent_embedder(agent_emb)                           # (B,D)
        x = torch.cat([context_enc, agent_enc, tgt_question], dim=-1)        # (B,2D+Q)
        return self.head(x)                                                  # (B,1)


def epoch_batch_loader(questions_by_agent, agents, batch_size, target_masks=None):
    """Yield batches of data"""

    permute_indices = torch.randperm(agents.shape[0])

    for i in range(0, agents.shape[0], batch_size):
        batch_indices = permute_indices[i:i+batch_size]

        batch_agents = agents[batch_indices]  # (B, A)

        batch_questions = [questions_by_agent[j] for j in batch_indices]
        if target_masks is not None:
            batch_masks = [target_masks[j] for j in batch_indices]
        else:
            batch_masks = [torch.ones(agent_questions.shape[0], dtype=torch.bool) for agent_questions in batch_questions]

        smallest_context = min([questions.shape[0] for questions in batch_questions]) # N, including target question

        if smallest_context == 1:
            # print(f"Smallest context size in this batch: {smallest_context}, batch_size {batch_size}")
            pass

        if smallest_context == 0:
            # print(f"Smallest context size in this batch: {smallest_context}")
            # print("HI")
            continue  # skip this batch if any agent has no questions or no valid target question

        if not all([mask.any().item() for mask in batch_masks]):
            # skip this batch if any agent has no valid target question
            continue

        batch_context = []
        batch_tgt_question = []
        batch_tgt_label = []
        for questions, mask in zip(batch_questions, batch_masks): # by agent
            # assert questions and mask have same length
            assert questions.shape[0] == mask.shape[0]

            # permute questions for this agents questions for random sampling
            perm = torch.randperm(questions.shape[0])  

            # get the first index where mask[perm] is True
            tgt_question_id = (mask[perm] == True).nonzero(as_tuple=True)[0][0].item()
            tgt_question = questions[perm][tgt_question_id][:-1]
            tgt_label = questions[perm][tgt_question_id][-1]

            batch_tgt_question.append(tgt_question)  # (Q,)
            batch_tgt_label.append(tgt_label)        # (1,)

            # remove all contexts that are in the target mask
            # TODO 
            # perm = perm[mask[perm] == False]
            # remove the target question from the context
            perm = perm[perm != tgt_question_id]     

            # TODO should I do padding instead of cutting off?
            if smallest_context - 1 == 0:
                # empty context
                context = questions[perm][:0]  # (0, Q+1)
            else:
                context = questions[perm][:smallest_context-1]  # (N-1, Q+1)
            batch_context.append(context)

        batch_context = torch.stack(batch_context).float().to(device=agents.device)            # (B, N-1, Q+1)
        batch_tgt_question = torch.stack(batch_tgt_question).float().to(device=agents.device)  # (B, Q)
        batch_tgt_label = torch.stack(batch_tgt_label).float().to(device=agents.device)        # (B, 1)

        yield batch_context, batch_agents, batch_tgt_question, batch_tgt_label


def train_epoch(model, questions_by_agent, agents, optimizer, criterion, batch_size=8):
    batchloader = epoch_batch_loader(questions_by_agent, agents, batch_size)
    model.train()

    tot_loss = 0
    accs = []
    for batch_context, batch_agents, batch_tgt_question, batch_tgt_label in batchloader:
        optimizer.zero_grad()

        outputs = model(batch_context, batch_agents, batch_tgt_question).squeeze()  # (B,)
        loss = criterion(outputs, batch_tgt_label.float().squeeze())                # (B,)

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).long()
        accuracy = (preds == batch_tgt_label.squeeze()).sum().item() / batch_tgt_label.size(0)

        accs.append(accuracy)
        tot_loss += loss.item()
    
    return tot_loss, sum(accs) / len(accs)

def evaluate(model, questions_by_agent, agents, masks=None):
    model.eval()
    with torch.no_grad():
        batchloader = epoch_batch_loader(questions_by_agent, agents, batch_size=1, target_masks=masks)
        all_preds = []
        all_labels = []
        for batch_context, batch_agents, batch_tgt_question, batch_tgt_label in batchloader:
            outputs = model(batch_context, batch_agents, batch_tgt_question).squeeze(1)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()

            all_preds.append(preds.cpu())
            all_labels.append(batch_tgt_label.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        accuracy = (all_preds == all_labels).sum().item() / all_labels.size(0)

    return accuracy 

def train_model(model, questions_by_agent, agents, config):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training'].get('weight_decay')
    )
    
    # # split data in train, val, test
    # if splitting by agent, then all questions for an agent go to the same split
    if config['data']['split_by'] == 'agent':
        num_agents = agents.shape[0]
        train_size = int(config['data']['train_ratio'] * num_agents)
        val_size = int(config['data']['val_ratio'] * num_agents)

        train_agents = agents[:train_size]
        val_agents = agents[train_size:train_size+val_size]
        test_agents = agents[train_size+val_size:]

        train_questions_by_agent = questions_by_agent[:train_size]
        val_questions_by_agent = questions_by_agent[train_size:train_size+val_size]
        test_questions_by_agent = questions_by_agent[train_size+val_size:]

    # if splitting by question, then questions for the same agent can go to different splits, and so we need to maintain the target masks
    elif config['data']['split_by'] == 'question':
        # all agents are used for training, validation and testing
        train_agents = agents
        val_agents = agents
        test_agents = agents

        train_questions_by_agent = []        
        val_questions_by_agent = []
        test_questions_by_agent = []

        val_masks = []
        test_masks = []

        tot_train_size = 0
        tot_val_size = 0
        tot_test_size = 0

        for agent_questions in questions_by_agent:
            num_questions = agent_questions.shape[0]

            if num_questions > 2:
                # If there's only one question, we can't create a validation or test set
                train_size = int(config['data']['train_ratio'] * num_questions)
                val_size = int(config['data']['val_ratio'] * num_questions)
            elif num_questions == 2:
                train_size = 1
                # randomly assign one to val or test
                val_size = 1 if torch.rand(1).item() > 0.5 else 0
                train_size += (1 - val_size)  # the other one goes to train
            else:  # num_questions == 1
                # randomly assign to train, val, or test
                rand_val = torch.rand(1).item()
                if rand_val < config['data']['train_ratio']:
                    train_size = 1
                    val_size = 0
                elif rand_val < config['data']['train_ratio'] + config['data']['val_ratio']:
                    train_size = 0
                    val_size = 1
                else:
                    train_size, val_size = 0, 0
            
            test_size = num_questions - train_size - val_size
            tot_train_size += train_size
            tot_val_size += val_size
            tot_test_size += test_size

            assert train_size + val_size <= num_questions

            # shuffle indices
            perm = torch.randperm(num_questions)

            train_questions = agent_questions[perm[:train_size]]

            val_questions = agent_questions[perm[:train_size+val_size]] # has access to training history
            val_mask = torch.zeros(val_questions.shape[0], dtype=torch.bool)
            val_mask[train_size:train_size+val_size] = True

            test_questions = agent_questions[perm] # has access to all questions
            test_mask = torch.zeros(test_questions.shape[0], dtype=torch.bool)
            test_mask[train_size+val_size:] = True

            train_questions_by_agent.append(train_questions)
            val_questions_by_agent.append(val_questions)
            val_masks.append(val_mask)
            test_questions_by_agent.append(test_questions)
            test_masks.append(test_mask)
        
        print(f"Average number of questions per agent in training set: {(sum([q.shape[0] for q in train_questions_by_agent])/len(train_questions_by_agent)):.2f}")
        print(f"Average number of questions per agent in validation set: {sum([q.shape[0] for q in val_questions_by_agent])/len(val_questions_by_agent):.2f}")
        print(f"Average number of questions per agent in test set: {sum([q.shape[0] for q in test_questions_by_agent])/len(test_questions_by_agent):.2f}")
        print(f"Total number of questions in training set: {tot_train_size}")
        print(f"Total number of questions in validation set: {tot_val_size}")
        print(f"Total number of questions in test set: {tot_test_size}")
        print(f"Actual proportions: train {tot_train_size/(tot_train_size+tot_val_size+tot_test_size):.2f}, val {tot_val_size/(tot_train_size+tot_val_size+tot_test_size):.2f}, test {tot_test_size/(tot_train_size+tot_val_size+tot_test_size):.2f}")
    else:
        raise ValueError("split_by must be 'agent' or 'question'")

    use_wandb = config['experiment'].get('use_wandb', False)

    if use_wandb:
        wandb.init(project=config['experiment']['project_name'],
            name=config['experiment']['experiment_name'] + f"_splitby{config['data']['split_by']}",
            config=config
        )

    val_accs = []
    test_accs = []

    for epoch in range(config['training']['epochs']):  # number of epochs
        optimizer.zero_grad()

        loss, train_accs = train_epoch(model, train_questions_by_agent, train_agents, optimizer, criterion, batch_size=config['training']['batchsize'])

        if config['data']['split_by'] == 'question':
            val_acc = evaluate(model, val_questions_by_agent, val_agents, val_masks)
            test_acc = evaluate(model, test_questions_by_agent, test_agents, test_masks)
        else:
            val_acc = evaluate(model, val_questions_by_agent, val_agents)
            test_acc = evaluate(model, test_questions_by_agent, test_agents)

        val_accs.append(val_acc)
        test_accs.append(test_acc)

        best_epoch, best_test_acc = get_best_epoch(val_accs, test_accs)

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': loss,
                'train_acc': train_accs,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'best/epoch': best_epoch,
                'best/test_acc': best_test_acc
            })

        if (epoch+1) % 10 == 0: print(f'Epoch [{epoch+1}/100], Loss: {loss:.4f}, Train Acc: {train_accs:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Best Test Acc: {best_test_acc:.4f} at epoch {best_epoch}')
    
    print(f'Best Val Epoch: {best_epoch}, Test Accuracy: {best_test_acc:.4f}')

    if use_wandb:
        wandb.finish()

    return best_test_acc

def get_best_epoch(val_accs, test_accs):
    best_val_idx = val_accs.index(max(val_accs))
    return best_val_idx, test_accs[best_val_idx]


def init_set_transformer(data: list[SetTransformerData]) -> TabularPredictor:
    device = get_torch_device()

    agent_question_scores = defaultdict(list)
    for agent, question, score in data:
        agent_question_scores[agent.id].append((question.embedding, score))

    question_emb_dim = len(data[0][1].embedding)
    agent_emb_dim = len(data[0][0].embedding)

    config = {
        'model': {
            'hidden_dim': 128,
            'n_heads': 4,
            'n_layers': 2
        },
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.0,
            'batchsize': 32
        },
        'data': {
            'split_by': 'question',
            'train_ratio': 0.7,
            'val_ratio': 0.15
        },
        'experiment': {
            'use_wandb': True,
            'project_name': 'agentsearch-set-transformer',
            'experiment_name': 'eval'
        }
    }

    model = TabularPredictor(
        question_embedding_dim=question_emb_dim,
        agent_embedding_dim=agent_emb_dim,
        hidden_dim=config['model']['hidden_dim'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers']
    ).to(device)

    unique_agents = list(agent_question_scores.keys())
    agents_tensor = torch.stack([
        torch.tensor(next(a.embedding for a in [agent for agent, _, _ in data if agent.id == aid]), dtype=torch.float32)
        for aid in unique_agents
    ]).to(device)

    questions_by_agent = []
    for aid in unique_agents:
        agent_data = agent_question_scores[aid]
        questions_tensor = torch.tensor([
            list(q_emb) + [score] for q_emb, score in agent_data
        ], dtype=torch.float32).to(device)
        questions_by_agent.append(questions_tensor)

    train_model(model, questions_by_agent, agents_tensor, config)

    return model


def set_transformer_match(
    history: list[SetTransformerData],
    model: TabularPredictor,
    question: Question,
    collection: str = "agents"
) -> list[Agent]:
    device = get_torch_device()
    model.eval()

    matches = Agent.match(question, top_k=8, collection=collection)
    agents = list(map(lambda m: m.agent, matches))

    agent_scores = []
    with torch.no_grad():
        for agent in agents:
            agent_history = [(q.embedding, score) for a, q, score in history if a.id == agent.id]

            if len(agent_history) > 0:
                context = torch.tensor([
                    list(q_emb) + [score] for q_emb, score in agent_history
                ], dtype=torch.float32).unsqueeze(0).to(device)
            else:
                context = torch.zeros((1, 0, len(question.embedding) + 1), dtype=torch.float32).to(device)

            agent_emb = torch.tensor(agent.embedding, dtype=torch.float32).unsqueeze(0).to(device)
            tgt_question = torch.tensor(question.embedding, dtype=torch.float32).unsqueeze(0).to(device)

            score = torch.sigmoid(model(context, agent_emb, tgt_question)).item()
            agent_scores.append((agent, score))

    agent_scores.sort(key=lambda x: x[1], reverse=True)
    return [agent for agent, _ in agent_scores]