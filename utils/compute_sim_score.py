import numpy as np
import os
from tqdm import tqdm

def open_file(path_to_txt):
    f = open(path_to_txt, 'r')
    txt = [el for el in f.read().split('\n') if el != '']
    f.close()
    return txt


def load_feature(path, pooling="mean"):

        features = np.loadtxt(path)

        if pooling == 'max':
            return np.amax(features, axis=0)
        elif pooling == 'mean':
            return np.mean(features, axis=0)
        else:
            raise ValueError('Please specify a pooling between max and mean.')

def evaluate_similarity_task(path_to_similarity_task, paths_to_dir_features, pooling, voices_code=['_A', '_C', '_H', '_J']):
                
    task = open_file(path_to_similarity_task)
    task = [el.split(' ') for el in task]
        
    scores_tot = []                                                                                                                                                   
    for test in tqdm(task):
        scores_test = []
        for voice_code in voices_code:

            paths_to_features_A = os.path.join(paths_to_dir_features, test[0] + voice_code + '.txt')
            A = load_feature(paths_to_features_A, pooling)

            paths_to_features_X = os.path.join(paths_to_dir_features, test[1] + voice_code + '.txt')
            X = load_feature(paths_to_features_X, pooling)

            simi_A_X = np.dot(A, X) / (np.linalg.norm(A) * np.linalg.norm(X))

            for i in range(2, len(test)):

                paths_to_features_B = os.path.join(paths_to_dir_features, test[i] + voice_code + '.txt')
                B = load_feature(paths_to_features_B, pooling)

                simi_B_X = np.dot(B, X) / (np.linalg.norm(B) * np.linalg.norm(X))

                if simi_A_X > simi_B_X:
                    scores_test.append(1)
                else:
                    scores_test.append(0)

        scores_test = np.mean(scores_test)
        scores_tot.append(scores_test)
    scores_tot = 100 * np.mean(scores_tot)
    
    print(f'The average score for these features and pooling for this task is {scores_tot:.1f}.')
    return scores_tot
    

#------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("eval_task", help="path to eval_task_csv")
    parser.add_argument("feature_dir", help="path to feature directory")
    parser.add_argument("output_path", help="path to output_file")
    parser.add_argument("--pooling",  help="pooling_type (mean or max)", type=str, default="mean")

    parser.parse_args()
    args, leftovers = parser.parse_known_args()

    

    score = evaluate_similarity_task(args.eval_task, args.feature_dir, args.pooling,  voices_code=['_A', '_C', '_H', '_J'])
    with open(args.output_path,'w') as outfile:
        outfile.write("{} pos with {} pooling".format(round(score,3), args.pooling))
