from typing import Dict
from datasets import load_dataset

def create_vocab(config: Dict):
    dataset = load_dataset("csv", data_files={
        "train": config['data']['train_path'],
        "test": config['data']['test_path']
    })
    
    word_counts = {}
    for split in dataset.values():
        for column in ['context', 'prompt', 'response']:
            try:
                for item in split[column]:
                    for word in item.split():
                        word_counts[word] = word_counts.get(word, 0) + 1
            except:
                pass

    vocab = list(dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)).keys())
    vocab.append("[unknown]")
    
    return vocab, word_counts