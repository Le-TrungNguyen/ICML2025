import pickle
import matplotlib.pyplot as plt
import os

class Perplexity:
    def __init__(self, layer_names = [], set_of_epsilons=[], perplexity=[], ranks=[], layer_mems=[], link='.'):
        self.layer_names = layer_names
        self.set_of_epsilons = set_of_epsilons
        self.perplexity = perplexity
        self.link = link 
        self.ranks = ranks
        self.layer_mems = layer_mems
        

    def plot(self, is_saved=False, name=None):
        if not self.set_of_epsilons or not self.perplexity:
            return

        # Vẽ đồ thị
        for layer_index, layer_perplexity in enumerate(self.perplexity):
            plt.plot(self.set_of_epsilons, layer_perplexity, label=f'Layer {layer_index + 1}')

        plt.xlabel('Epsilon')
        plt.ylabel('Perplexity')
        plt.title('Perplexity vs Epsilon for Different Layers')
        plt.legend()
        plt.grid(True)

        plt.xticks(self.set_of_epsilons)

        if is_saved:
            if name is None: name = 'perplexity.svg'
            file_path = os.path.join(self.link, name)
            plt.savefig(file_path)
            print(f'Figure is saved at {file_path}')

        plt.show()

    def save(self, link):
        with open(link, 'wb') as file:
            pickle.dump({
                'layer_names': self.layer_names,
                'set_of_epsilons': self.set_of_epsilons,
                'perplexity': self.perplexity,
                'ranks': self.ranks,
                'layer_mems': self.layer_mems
            }, file)
        print(f'Perplexity is saved at {link}')

    def load(self, link):
        with open(link, 'rb') as file:
            data = pickle.load(file)
            self.layer_names = data['layer_names']
            self.set_of_epsilons = data['set_of_epsilons']
            self.perplexity = data['perplexity']
            self.ranks = data['ranks']
            self.layer_mems = data['layer_mems']
    
    def get_suitable_ranks(self, best_indices, num_of_finetuned):
        if num_of_finetuned == None or num_of_finetuned > len(self.layer_mems):
            print("[Perplexity class] Warning, num_of_finetuned is bigger than total number of layer or None, set it to be total number of layer")
            num_of_finetuned = len(self.layer_mems)

        suitable_ranks = []
        start_layer = len(self.layer_mems) - num_of_finetuned

        rank_idx = 0
        for layer_idx in range(start_layer, len(self.layer_mems)):
            suitable_ranks.append(self.ranks[layer_idx][best_indices[rank_idx]])
            rank_idx += 1
        
        return suitable_ranks
    
    def get_suitable_mems(self, best_indices, num_of_finetuned):
        if num_of_finetuned == None or num_of_finetuned > len(self.layer_mems):
            print("[Perplexity class] Warning, num_of_finetuned is bigger than total number of layer or None, set it to be total number of layer")
            num_of_finetuned = len(self.layer_mems)

        suitable_mems = []
        start_layer = len(self.layer_mems) - num_of_finetuned

        rank_idx = 0
        for layer_idx in range(start_layer, len(self.layer_mems)):
            suitable_mems.append(self.layer_mems[layer_idx][best_indices[rank_idx]])
            rank_idx += 1
        
        return suitable_mems

    def find_best_combination(self, budget, num_of_finetuned=None):
        if num_of_finetuned == None or num_of_finetuned > len(self.layer_mems):
            print("[Perplexity class] Warning, num_of_finetuned is bigger than total number of layer or None, set it to be total number of layer")
            num_of_finetuned = len(self.layer_mems)
        
        total_layer = len(self.layer_mems)
        total_rank = len(self.layer_mems[0])

        best_memory = 0
        best_perplexity = float('inf')
        best_indices = []

        min_memory = float('inf')
        min_memory_indices = []

        def backtrack(layer, current_memory, current_perplexity, indices):
            nonlocal best_memory, best_perplexity, best_indices, min_memory, min_memory_indices

            if layer == total_layer:
                if current_memory <= budget and current_perplexity < best_perplexity:
                    best_memory = current_memory
                    best_perplexity = current_perplexity
                    best_indices = indices[:]
                
                if current_memory < min_memory:
                    min_memory = current_memory
                    min_memory_indices = indices[:]

                return

            for rank in range(total_rank):
                new_memory = current_memory + self.layer_mems[layer][rank]
                new_perplexity = current_perplexity + self.perplexity[layer][rank]

                indices.append(rank)
                backtrack(layer + 1, new_memory, new_perplexity, indices)
                indices.pop()

        start_layer = len(self.layer_mems) - num_of_finetuned
        backtrack(start_layer, 0, 0, [])

        if best_perplexity == float('inf'):
            print("Warning: No valid combination found within the budget. Returning the combination with the smallest memory.")
            best_perplexity = 0
            i = 0
            for layer in range(start_layer, len(self.layer_mems)):
                best_perplexity += self.perplexity[layer][min_memory_indices[i]]
                i+= 1
            return min_memory, best_perplexity, min_memory_indices, self.get_suitable_ranks(min_memory_indices, num_of_finetuned)
        else:
            return best_memory, best_perplexity, best_indices, self.get_suitable_ranks(best_indices, num_of_finetuned)

def merged_perplexity(*links_to_perplexity):

    if not links_to_perplexity:
        raise ValueError("No perplexity files provided!")

    perplexity_merged = Perplexity()
    perplexity_merged.load(links_to_perplexity[0])

    for link in links_to_perplexity[1:]:
        perplexity_temp = Perplexity()
        perplexity_temp.load(link)

        perplexity_merged.set_of_epsilons += perplexity_temp.set_of_epsilons
        perplexity_merged.perplexity = [row1 + row2 for row1, row2 in zip(perplexity_merged.perplexity, perplexity_temp.perplexity)]
        perplexity_merged.ranks = [row1 + row2 for row1, row2 in zip(perplexity_merged.ranks, perplexity_temp.ranks)]
        perplexity_merged.layer_mems = [row1 + row2 for row1, row2 in zip(perplexity_merged.layer_mems, perplexity_temp.layer_mems)]

    saved_location = os.path.dirname(links_to_perplexity[0])
    os.makedirs(saved_location, exist_ok=True)
    saved_file = os.path.join(saved_location, 'perplexity_combined.pkl')
    
    perplexity_merged.save(saved_file)

    return saved_file