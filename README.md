#Phase 0 log
Description: Phase 0 is to test whether a surrogate can cleanly read an image and understand the image content correctly. 
Modules:
1. The phase0_test.py: This one is to quickly test how well the model can do full transcription and answer the necessary query on n image sample, split by category in the UI domain (banking, communication etc). 
2. The phase0_eval.py: This one runs the phase0_test on an evaluation level. It logs the metrics of * binding_acc = this is how well a model can answer the question(s)
* token_presence = this is also using the question(s) as the goal but instead of directly asking the model and comparing the answer (that is the binding_acc), it searches in the full transcription to see if the correct answer is present
* content_fidelity = this measures the ROUGE-L score for full transcription. The idea is to see how well a model can extract the information

#Phase 1 log
Description: Phase 1 is to test ce_loss, align_loss and salience map for when the model is prompted with a targeted question vs full transcription. 
1. ce_loss: I need to verify that the ce_loss on clean image and ground truth full text must be low. That means a model assigns a high logprob of generating the full ground truth text given the image. If it is already high or if it is also low on a decoy image, then the model's ce_loss is questionable
2. align_loss: pending
3. salience_map: the idea is that when the model is answering a target question, like 'who is the account holder', the attention should be focused on the 'ACCOUNT HOLDER' and 'Ella Thompson' area for example, and not completely random. It should also be different when it is asked to do a full transcription.  
