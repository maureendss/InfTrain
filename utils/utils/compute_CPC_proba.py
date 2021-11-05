import os
from time import time
import torch
import torchaudio
from cpc.feature_loader import loadModel, getCheckpointData
from cpc.train import getCriterion

def loadCriterion(pathCheckpoint, downsampling, nSpeakers, nPhones):
    _, _, locArgs = getCheckpointData(os.path.dirname(pathCheckpoint))
    criterion = getCriterion(locArgs, downsampling, nSpeakers, nPhones)

    state_dict = torch.load(pathCheckpoint, 'cpu')

    # for newer versions of CPC, the name is changed
    try:
        criterion.load_state_dict(state_dict["cpcCriterion"])
    except RuntimeError:
        state_dict["cpcCriterion"]['speakerEmb.weight'] = state_dict["cpcCriterion"]['speaker_norm.emb.weight']
        del state_dict["cpcCriterion"]['speaker_norm.emb.weight']
        criterion.load_state_dict(state_dict["cpcCriterion"])

    return criterion

def getPositiveSamples(encodedData, nPredicts=12):
    batchSize, nNegativeExt, dimEncoded = encodedData.size()
    outputs = []

    for k in range(1, nPredicts + 1):
        # Positive samples
        if k < nPredicts:
            posSeq = encodedData[:, k:-(nPredicts-k)]
        else:
            posSeq = encodedData[:, k:]

        posSeq = posSeq.view(batchSize, 1, posSeq.size(1), dimEncoded)
        outputs.append(posSeq)

    return outputs

def scoring(logprobs):
    n = len(logprobs)//2
    true_preds = 0
    results = []
    for i in range(n):
        if logprobs[2*i] > logprobs[2*i + 1]:
            true_preds += 1
            results.append(True)
        else:
            results.append(False)
    if n != 0:
        print("Test accuracy: {}/{} ({:.2f}%)".format(true_preds, n, 100*true_preds/n))



def compute_score_CPC(seqPath,
                      cpcModel,
                      cpcCriterion,
                      speakerLabel=0,
                      nTemporal=12,
                      logits_scaling=1,
                      reduce_method='sum',
                      prob_estimator='loss',
                      n_negative_sampling=64, 
                      average_total=False
                      ):
    '''
    Comment on some useful args:
        logits_scaling:  put this high to avoid having 0. log proba for near temporal steps when
                         using sigmoid, but it seems that 1 (default) gives the best results
        reduce_method:  'sum' seems to work best
        prob_estimator:  using 'sigmoid' is faster as we don't need to compute negative samples,
                         but using 'negative_sampling' seems to have better results as this is
                         the way the CPC model is trained (however this will make the scores varying)
   n_negative_sampling:  number of negative sample, default to 8
    '''
    assert reduce_method in ['sum', 'mean']
    assert prob_estimator in ['sigmoid', 'negative_sampling', 'loss']
    with torch.no_grad():
        # Read the input signals
        seq = torchaudio.load(seqPath)[0]  # 1 x frames
        seq = seq[:, :].view(1, 1, -1).cuda()  # 1 x 1 x frames

        # Read CPC features
        cpcModel.gAR.hidden = None
        cFeature, encodedData, label = cpcModel(seq, label=None)

        ## cFeature: 1 x T x D_feat
        ## encodedData: 1 x T x D_enc

        # Prepare CPC features for criterion
        batchSize, seqSize, _ = cFeature.size()
        windowSize = seqSize - cpcCriterion.nPredicts  # T - 12
        cFeature = cFeature[:, :windowSize]  # 1 x (T - 12) x D_feats

        # Get positive encoded samples
        if prob_estimator == 'negative_sampling' or prob_estimator == 'loss':
            if n_negative_sampling is not None:
                cpcCriterion.negativeSamplingExt = n_negative_sampling

            sampledData, labelLoss = cpcCriterion.sampleClean(encodedData,
                                                              windowSize)  # 12 x 1 x (1 + n_negative_sampling) x (T - 12) x D_enc
        else:
            sampledData = getPositiveSamples(encodedData, cpcCriterion.nPredicts)  # 12 x 1 x 1 x (T - 12) x D_enc

        # Speaker embeddings
        if getattr(cpcCriterion, 'speakerEmb', None) is not None:
            label = torch.tensor(speakerLabel).cuda()
            l_ = label.view(batchSize, 1).expand(batchSize, windowSize)  # 1 x (T - 12)
            embeddedSpeaker = cpcCriterion.speakerEmb(l_)  # 1 x (T - 12) x D_spkemb
            cFeature = torch.cat([cFeature, embeddedSpeaker], dim=2)  # 1 x (T - 12) x (D_feat+D_spkemb)

        # Compute the criterion outputs
        predictions = cpcCriterion.wPrediction(cFeature, sampledData)  # 12 x 1 x 1 x (T - 12)

        # Compute the pseudo log-probas
        lp_score = 0.
        outLosses = [0 for x in range(nTemporal)]
        outAcc = [0 for x in range(nTemporal)]
        for k, outputs in enumerate(predictions[:nTemporal]):
            if prob_estimator == 'sigmoid':
                logits = outputs[0] / logits_scaling
                logits = logits.sigmoid()
            elif prob_estimator == 'negative_sampling':
                logits = outputs[0] / logits_scaling
                logits = logits.softmax(0)
            elif prob_estimator == 'loss':
                outputs = outputs.permute(0, 2, 1)
                outputs = outputs.contiguous().view(-1, outputs.size(2))
                lossK = cpcCriterion.lossCriterion(outputs, labelLoss)

                outLosses[k] += lossK.view(1, -1)
                _, predsIndex = outputs.max(1)
                outAcc[k] += torch.sum(predsIndex == labelLoss).float().view(1, -1).item()

            # if reduce_method == 'sum':
            #     lp_score += logits[0].log().sum()
            # elif reduce_method == 'mean':
            #     lp_score += logits[0].log().mean()
            #lp_score /= nTemporal

            # logits = outputs[0] / logits_scaling
            # if logits.size(0) == 1:
            #     logits = logits.sigmoid()
            # else:
            #     logits = logits.softmax(0)
            # if reduce_method == 'sum':
            #     lp_score += logits[0].log().sum()
            # elif reduce_method == 'mean':
            #     lp_score += logits[0].log().mean()
            # lp_score /= nTemporal
        outLosses = torch.FloatTensor(outLosses).cpu() / (windowSize * batchSize)
        outAcc = torch.FloatTensor(outAcc).cpu() / (windowSize * batchSize)
        
    if average_total:
        outLosses = outLosses.mean().item()
        outAcc = outAcc.mean().item()
            

    return outLosses, outAcc

# def compute_score_CPC(seqPath,
#                       cpcModel,
#                       cpcCriterion,
#                       speakerLabel=0,
#                       nTemporal=12,
#                       logits_scaling=1,
#                       reduce_method='sum',
#                       prob_estimator='negative_sampling',
#                       n_negative_sampling=None, score="logprob"):
#     '''
#     Comment on some useful args:
#         logits_scaling:  put this high to avoid having 0. log proba for near temporal steps when
#                          using sigmoid, but it seems that 1 (default) gives the best results
#          reduce_method:  'sum' seems to work best
#         prob_estimator:  using 'sigmoid' is faster as we don't need to compute negative samples,
#                          but using 'negative_sampling' seems to have better results as this is
#                          the way the CPC model is trained (however this will make the scores varying)
#    n_negative_sampling:  leave this to 'None' and the model will use 128(defaut) negative samples
#     score:                either [accuracy] (mean on 12 frames) or [logprob]
#     '''
#     assert reduce_method in ['sum', 'mean']
#     assert prob_estimator in ['sigmoid', 'negative_sampling']
#     with torch.no_grad():
#         # Read the input signals
#         seq = torchaudio.load(seqPath)[0] # 1 x frames
#         seq = seq[:,:].view(1, 1, -1).cuda() # 1 x 1 x frames

#         # Read CPC features
#         cpcModel.gAR.hidden = None
#         cFeature, encodedData, label = cpcModel(seq, label=None)
#         ## cFeature: 1 x T x D_feat
#         ## encodedData: 1 x T x D_enc

#         # Prepare CPC features for criterion
#         batchSize, seqSize, _ = cFeature.size()
#         windowSize = seqSize - cpcCriterion.nPredicts # T - 12
#         cFeature = cFeature[:, :windowSize] # 1 x (T - 12) x D_feat

#         # Get positive encoded samples
#         if prob_estimator=='negative_sampling':
#             if n_negative_sampling is not None:
#                 cpcCriterion.negativeSamplingExt = n_negative_sampling
#             sampledData, _ = cpcCriterion.sampleClean(encodedData, windowSize) # 12 x 1 x (1 + n_negative_sampling) x (T - 12) x D_enc
#         else:
#             sampledData = getPositiveSamples(encodedData, cpcCriterion.nPredicts) # 12 x 1 x 1 x (T - 12) x D_enc

#         # Speaker embeddings
#         if cpcCriterion.speakerEmb is not None:
#             label = torch.tensor(speakerLabel).cuda()
#             l_ = label.view(batchSize, 1).expand(batchSize, windowSize) # 1 x (T - 12)
#             embeddedSpeaker = cpcCriterion.speakerEmb(l_) # 1 x (T - 12) x D_spkemb
#             cFeature = torch.cat([cFeature, embeddedSpeaker], dim=2) # 1 x (T - 12) x (D_feat+D_spkemb)

#         # Compute the criterion outputs
#         predictions = cpcCriterion.wPrediction(cFeature, sampledData) # 12 x 1 x 1 x (T - 12)

#         if score == "accuracy":
#             acc_score = 0
#             for outputs in predictions[:nTemporal]:
#                 acc_score += outputs[0][0].mean() #no neg samples

#             acc_score /= nTemporal
#             out_score = acc_score.item()


#         # Compute the pseudo log-probas
#         if score == "logprob":
#             lp_score = 0.
#             for outputs in predictions[:nTemporal]:
#                 logits = outputs[0]/logits_scaling
#                 if logits.size(0) == 1:
#                     logits = logits.sigmoid()
#                 else:
#                     logits = logits.softmax(0)
#                 if reduce_method == 'sum':
#                     lp_score += logits[0].log().sum()
#                 elif reduce_method == 'mean':
#                     lp_score += logits[0].log().mean()
#             lp_score  /= nTemporal
#             out_score = lp_score.item()

#     return out_score

