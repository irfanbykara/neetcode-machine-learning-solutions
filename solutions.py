#### Answer for the first question. 

class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        for iteration in range(iterations):

            init -= learning_rate * 2 * init
        return round(init, 5)

#### Answer for the second question. 
import numpy as np
from numpy.typing import NDArray


# Helpful functions:
# https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
# https://numpy.org/doc/stable/reference/generated/numpy.mean.html
# https://numpy.org/doc/stable/reference/generated/numpy.square.html

class Solution:
    
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:

        res = np.matmul(X,weights)
        return np.round(res,5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        error = model_prediction - ground_truth 
        squared_error = np.square(error)
        mean_squared_error = np.mean(squared_error)
        return np.round(mean_squared_error,5)


#### Answer for the third question. 
import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        learning_rate = 0.01
        weights = initial_weights
        for iteration in range(num_iterations):
            prediction = self.get_model_prediction(X,weights)
            for i,weight in enumerate(weights):
                derivative = self.get_derivative(prediction,Y,len(X),X,i)
                weights[i] -= derivative * learning_rate
        # you will need to call get_derivative() for each weight
        # and update each one separately based on the learning rate!
        # return np.round(your_answer, 5)
        return np.round(weights,5)

###Answer for the forth question:
import torch
import torch.nn
from torchtyping import TensorType

# Helpful functions:
# https://pytorch.org/docs/stable/generated/torch.reshape.html
# https://pytorch.org/docs/stable/generated/torch.mean.html
# https://pytorch.org/docs/stable/generated/torch.cat.html
# https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html

# Round your answers to 4 decimal places using torch.round(input_tensor, decimals = 4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # torch.reshape() will be useful - check out the documentation
        return torch.reshape(to_reshape,((to_reshape.shape[0]*to_reshape.shape[1])//2,2))

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # torch.mean() will be useful - check out the documentation
        return torch.mean(to_avg,dim=0)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # torch.cat() will be useful - check out the documentation
        return torch.cat((cat_one,cat_two),1)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # torch.nn.functional.mse_loss() will be useful - check out the documentation
        return torch.nn.functional.mse_loss(prediction,target)

###Answer for the fifth question:
import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(784,512)
        self.final = nn.Linear(512,10)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # Define the architecture here 
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        flattened = images.view(images.shape[0],28*28)
        out = self.linear(flattened)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.final(out)
        out = self.sigmoid(out)
        return torch.round(out,decimals=4)
        # Return the model's prediction to 4 decimal places

###Answer for the sixth question:
import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        all_positive_words = []
        all_negative_words = []
        max_len = 0
        for pos in positive:
            if len(pos.split(" ")) > max_len:
                max_len = len(pos.split(" "))
            for word in pos.split(" "):
            
                all_positive_words.append(word)

        for neg in negative:
            if len(neg.split(" ")) > max_len:
                max_len = len(neg.split(" "))
            for word in neg.split(" "):
                all_negative_words.append(word)
        
        all_words = all_positive_words + all_negative_words
        all_words_set = set(all_words)
        
        my_dict = dict.fromkeys(sorted(all_words_set))
        x= 1
        for key,val in my_dict.items():
            my_dict[key] = x
            x += 1
        all_integer = []

        for pos in positive:
            sub_list = []
            for word in pos.split(" "):
                
                sub_list.append(my_dict[word])
            while max_len > len(sub_list):
                sub_list.append(0)
            all_integer.append(sub_list)
    
            
        for neg in negative:
            sub_list = []
            for word in neg.split(" "):
                sub_list.append(my_dict[word])
            while max_len > len(sub_list):
                sub_list.append(0)

            all_integer.append(sub_list)
        return torch.tensor(all_integer)

###Answer for the seventh question:
import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding = nn.Embedding(embedding_dim=16,num_embeddings=vocabulary_size)
        self.linear = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        embedded = self.embedding(x)
        out = embedded.mean(dim=1)
        out = self.linear(out)
        out = self.sigmoid(out)
        out = torch.round(out,decimals=4)
        return out

###Answer for the eigth question
import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)
        raw_dataset_list = raw_dataset.split(" ")
        len_dataset = len(raw_dataset_list)
        random_indices = torch.randint(0, len_dataset - context_length, (batch_size,))
        X = []
        Y = []
        for index in random_indices:
            temp_x = []
            temp_y = []
            temp_x = raw_dataset_list[index:index+context_length]
            temp_y = raw_dataset_list[index+1:index+context_length+1]
            X.append(temp_x)
            Y.append(temp_y)
        return X,Y
            


### Answer for the ninth question:
import torch
import torch.nn as nn
from torchtyping import TensorType
import torch.nn.functional as F

# 0. Instantiate the linear layers in the following order: Key, Query, Value.
# 1. Biases are not used in Attention, so for all 3 nn.Linear() instances, pass in bias=False.
# 2. torch.transpose(tensor, 1, 2) returns a B x T x A tensor as a B x A x T tensor.
# 3. This function is useful:
#    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
# 4. Apply the masking to the TxT scores BEFORE calling softmax() so that the future
#    tokens don't get factored in at all.
#    To do this, set the "future" indices to float('-inf') since e^(-infinity) is 0.
# 5. To implement masking, note that in PyTorch, tensor == 0 returns a same-shape tensor 
#    of booleans. Also look into utilizing torch.ones(), torch.tril(), and tensor.masked_fill(),
#    in that order.
class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.k0 = nn.Linear(embedding_dim,attention_dim,bias=False)
        self.q0 = nn.Linear(embedding_dim,attention_dim,bias=False)
        self.v0 = nn.Linear(embedding_dim,attention_dim,bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Return your answer to 4 decimal places
        k = self.k0(embedded)
        q = self.q0(embedded)
        v = self.v0(embedded)

        attn = torch.matmul(q,torch.transpose(k,1,2)) / self.attention_dim ** 0.5
        mask = torch.tril(torch.ones(embedded.shape[1], embedded.shape[1])).unsqueeze(0).repeat(embedded.shape[0], 1, 1)
        attention_scores = attn.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        res = torch.matmul(attention_weights,v)
        return torch.round(res,decimals=4)

### Answer for the tenth question:
import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        # Hint: nn.ModuleList() will be useful. It works the same as a Python list
        # but is useful here since instance variables of any subclass of nn.Module
        # must also be subclasses of nn.Module

        # Use self.SingleHeadAttention(embedding_dim, head_size) to instantiate. You have to calculate head_size.

        super().__init__()
        torch.manual_seed(0)
        self.embedding_dim = embedding_dim 
        self.head_size = attention_dim // num_heads
        self.num_heads = num_heads
        self.layers = nn.ModuleList([self.SingleHeadAttention(embedding_dim,self.head_size) for _ in range(num_heads)])
        

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        
        attns = []
        for layer in self.layers:
            attn = layer(embedded)
            print(attn.shape)
            attns.append(attn)
        return torch.cat(attns,dim=-1)

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        
        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim = 2)

            return scores @ v

### Answer for the eleventh question:
import torch
import torch.nn as nn
from torchtyping import TensorType

# Even though the original diagram created by Google 
# has "Norm" after Attention in the bottom component, and 
# "Norm" after FeedForward in the top component, Norm should
# be applied first in both cases (before Attention & before FeedForward),
# and in each case, the output (specifically the output of attention
# in the first case & output of FeedForward in the second case) should
# be added to the tensor passed in to Norm. Researchers have found this
# architecture to be superior for LLM performance.
class TransformerBlock(nn.Module):
    
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.multi_headed_attention = self.MultiHeadedSelfAttention(model_dim,num_heads)
        self.norm = nn.LayerNorm(model_dim)
        self.feed_forward = self.VanillaNeuralNetwork(model_dim)
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Round answer to 4 decimal places
        torch.manual_seed(0)
        out = self.norm(embedded)
        out = self.multi_headed_attention(out)
        temp = embedded + out 
        out = self.norm(temp)
        out = self.feed_forward(out)
        out = out + temp
        return torch.round(out,decimals=4)

    class MultiHeadedSelfAttention(nn.Module):

        class SingleHeadAttention(nn.Module):
            def __init__(self, model_dim: int, head_size: int):
                super().__init__()
                torch.manual_seed(0)
                self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                self.value_gen = nn.Linear(model_dim, head_size, bias=False)
            
            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                k = self.key_gen(embedded)
                q = self.query_gen(embedded)
                v = self.value_gen(embedded)

                scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
                context_length, attention_dim = k.shape[1], k.shape[2]
                scores = scores / (attention_dim ** 0.5)

                lower_triangular = torch.tril(torch.ones(context_length, context_length))
                mask = lower_triangular == 0
                scores = scores.masked_fill(mask, float('-inf'))
                scores = nn.functional.softmax(scores, dim = 2)

                return scores @ v
            
        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.att_heads = nn.ModuleList()
            for i in range(num_heads):
                self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            head_outputs = []
            for head in self.att_heads:
                head_outputs.append(head(embedded))
            concatenated = torch.cat(head_outputs, dim = 2)
            return concatenated
    
    class VanillaNeuralNetwork(nn.Module):

        def __init__(self, model_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.up_projection = nn.Linear(model_dim, model_dim * 4)
            self.relu = nn.ReLU()
            self.down_projection = nn.Linear(model_dim * 4, model_dim)
            self.dropout = nn.Dropout(0.2) # using p = 0.2
        
        def forward(self, x: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            return self.dropout(self.down_projection(self.relu(self.up_projection(x))))


###Answer for the twelveth question:
import torch
import torch.nn as nn
from torchtyping import TensorType

# 1. Remember to include an additional LayerNorm after the block sequence and before the final linear layer
# 2. Instantiate in the following order: Word embeddings, position embeddings, transformer blocks, final layer norm, and vocabulary projection.
class GPT(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Word embeddings (vocab_size → model_dim)
        self.token_embedding = nn.Embedding(vocab_size, model_dim)

        self.position_embedding = nn.Embedding(context_length, model_dim)

        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(self.TransformerBlock(model_dim,num_heads))
        # Hint: nn.Sequential() will be useful for the block sequence
        self.blocks = nn.Sequential(*self.blocks)
        self.final_norm = nn.LayerNorm(model_dim)

        self.w_o = nn.Linear(model_dim,vocab_size)
        


    def forward(self, context: TensorType[int]) -> TensorType[float]:
        context = context.to(torch.long)  # for embedding
        B, T = context.shape

        token_emb = self.token_embedding(context)  # (B, T, model_dim)
        positions = torch.arange(T, device=context.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(positions)  # (1, T, model_dim)

        x = token_emb + pos_emb  # (B, T, model_dim)
        x = x.to(torch.float32)  # ensure compatibility

        x = self.blocks(x)
        x = self.final_norm(x)
        x = self.w_o(x)
        print(x.shape)
        x = nn.functional.softmax(x, dim=-1)
        return torch.round(x, decimals=4)
        
        

    # Do NOT modify the code below this line
    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    torch.manual_seed(0)
                    self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.value_gen = nn.Linear(model_dim, head_size, bias=False)
                
                def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                    k = self.key_gen(embedded)
                    q = self.query_gen(embedded)
                    v = self.value_gen(embedded)

                    scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
                    context_length, attention_dim = k.shape[1], k.shape[2]
                    scores = scores / (attention_dim ** 0.5)

                    lower_triangular = torch.tril(torch.ones(context_length, context_length))
                    mask = lower_triangular == 0
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim = 2)

                    return scores @ v
                
            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                torch.manual_seed(0)
                self.att_heads = nn.ModuleList()
                for i in range(num_heads):
                    self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                head_outputs = []
                for head in self.att_heads:
                    head_outputs.append(head(embedded))
                concatenated = torch.cat(head_outputs, dim = 2)
                return concatenated
        
        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2) # using p = 0.2
            
            def forward(self, x: TensorType[float]) -> TensorType[float]:
                torch.manual_seed(0)
                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.linear_network = self.VanillaNeuralNetwork(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            embedded = embedded + self.attention(self.first_norm(embedded)) # skip connection
            embedded = embedded + self.linear_network(self.second_norm(embedded)) # another skip connection
            return embedded

### Answer for the thirteenth question:
import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        generator = torch.Generator()
        generator.manual_seed(0)
        initial_state = generator.get_state()
        res = []

        # Ensure context is 1D LongTensor
        context = context.to(dtype=torch.long).flatten()

        for i in range(new_chars):
            # Prepare input for the model: shape [1, context_length]
            input_tensor = context[-context_length:].unsqueeze(0)

            logits = model(input_tensor)               # [1, context_length, vocab_size]
            last_logits = logits[:, -1, :]             # [1, vocab_size]
            probs = torch.softmax(last_logits, dim=-1) # [1, vocab_size]

            next_int = torch.multinomial(probs, num_samples=1, generator=generator)  # [1, 1]
            next_char = int_to_char[next_int.item()]

            generator.set_state(initial_state)
            res.append(next_char)

            # Update context with new token
            context = torch.cat([context, next_int.squeeze(0)], dim=0)[-context_length:]

        return "".join(res)





