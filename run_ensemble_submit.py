import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from common import *
from bms import *

from lib.net.lookahead import *
from lib.net.radam import *

#from dataset_224 import *
from dataset_ensemble import *

#from fairseq_model import *
from fairseq_model_ensemble import *


# ----------------
is_mixed_precision = True #False  #


###################################################################################################
import torch.cuda.amp as amp
# if is_mixed_precision:
#     class AmpNet(Net):
#         @torch.cuda.amp.autocast()
#         def forward(self, image):
#             #return super(AmpNet, self).forward(image)
#             return super(AmpNet, self).forward_argmax_decode(image)
# else:
#     AmpNet = Net
AmpNet_448 = Net_448
AmpNet_512 = Net_512

# start here ! ###################################################################################

def fast_remote_unrotate_augment(r):
    image_448 = r['image_448']
    image_512 = r['image_512']

    index = r['index']
    h, w = image_448.shape

    # if h > w:
    #     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    l= r['d'].orientation
    if l == 1:
        image_448 = np.rot90(image_448, -1)
        image_512 = np.rot90(image_512, -1)

    if l == 2:
        image_448 = np.rot90(image_448, 1)
        image_512 = np.rot90(image_512, 1)

    if l == 3:
        image_448 = np.rot90(image_448, 2)
        image_512 = np.rot90(image_512, 2)

    #image = cv2.resize(image, dsize=(image_size,image_size), interpolation=cv2.INTER_LINEAR)
    #assert (image_size==448)


    image_448 = image_448.astype(np.float16) / 255
    image_512 = image_512.astype(np.float16) / 255

    image_448 = torch.from_numpy(image_448).unsqueeze(0).repeat(3,1,1)
    image_512 = torch.from_numpy(image_512).unsqueeze(0).repeat(3,1,1)

    #r={}
    r['image_448'] = image_448
    r['image_512'] = image_512

    return r


def do_predict(net_448,net_512, tokenizer, valid_loader):

    text = []

    start_timer = timer()
    valid_num = 0
    for t, batch in enumerate(valid_loader):
        print(batch)
        batch_size = len(batch['image_448'])
        image_448 = batch['image_448'].cuda()
        image_512 = batch['image_512'].cuda()
        token = batch['token'].cuda()
        length = batch['length']
        net_448.eval()
        net_512.eval()
        
        with torch.no_grad():
            with amp.autocast():
                #k = net.forward_argmax_decode(image)
                    
                
                logit1 = net_448(image_448, token, length)
                logit2 = net_512(image_512, token, length)
                logit = (logit1+logit2)/2
                k = logit.argmax(-1)

                k = k.data.cpu().numpy()
                k = tokenizer.predict_to_inchi(k)
                text.extend(k)

        valid_num += batch_size
        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), time_to_str(timer() - start_timer, 'sec')),
              end='', flush=True)

    assert(valid_num == len(valid_loader.dataset))
    print('')
    return text




def run_submit():
    gpu_no = int(os.environ['CUDA_VISIBLE_DEVICES'])


    fold = 3
    out_dir = \
        'C:/Users/Kaggle/BMS/heng_clean/submit/ensemble_test/fold%d' % fold
    initial_checkpoint_448 = 'C:/Users/Kaggle/BMS/heng_clean/output/fold3_LR_0.00001_sz_448/checkpoint/swa_fold3.pth'
    initial_checkpoint_512 = 'C:/Users/Kaggle/BMS/heng_clean/output/fold3_LR_0.000005_sz_512/checkpoint/swa_fold3.pth'
           
    is_norm_ichi = False #True
    if 1:

        ## setup  ----------------------------------------
        mode = 'local'
        #mode = 'remote'

        submit_dir = out_dir + '/valid/%s-%s-gpu%d'%(mode, initial_checkpoint_448[-18:-4],gpu_no)
        os.makedirs(submit_dir, exist_ok=True)

        log = Logger()
        log.open(out_dir + '/log.submit.txt', mode='a')
        log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
        log.write('is_norm_ichi = %s\n' % is_norm_ichi)
        log.write('\n')

        #
        ## dataset ------------------------------------
        tokenizer = load_tokenizer()

        if 'remote' in mode: #1_616_107
            df_valid = make_fold('test')
            print("Running predict for {} rows".format(len(df_valid)))
            #if gpu_no==0 :  df_valid = df_valid[        : 400_000]
            #if gpu_no==1 :  df_valid = df_valid[ 400_000: 800_000]
            #if gpu_no==2 :  df_valid = df_valid[ 800_000:1200_000]
            #if gpu_no==3 :  df_valid = df_valid[1200_000]
            

        if 'local' in mode:  #484_837
            df_train, df_valid = make_fold('train-%d' % fold)
            
            #df_valid = df_valid[:40_000]
            #df_valid = df_valid[:5000]
            df_valid = df_valid[:10_000]

        df_valid = df_valid.sort_values('length').reset_index(drop=True)
        valid_dataset = BmsDataset_448_512(df_valid, tokenizer, augment=fast_remote_unrotate_augment)
        valid_loader  = DataLoader(
            valid_dataset,
            sampler = SequentialSampler(valid_dataset),
            batch_size  = 64, #*6,
            drop_last   = False,
            num_workers = 0,
            pin_memory  = True,
            #collate_fn  = lambda batch: null_collate(batch,False),
        )
        log.write('mode : %s\n'%(mode))
        log.write('valid_dataset : \n%s\n'%(valid_dataset))

        ## net ----------------------------------------
        if 1:
            tokenizer = load_tokenizer()
            net_448 = AmpNet_448().cuda()
            net_512 = AmpNet_512().cuda()

            net_448.load_state_dict(torch.load(initial_checkpoint_448)['state_dict'], strict=True)
            net_448 = torch.jit.script(net_448)

            net_512.load_state_dict(torch.load(initial_checkpoint_512)['state_dict'], strict=True)
            net_512 = torch.jit.script(net_512)


            #---
            start_timer = timer()
            predict = do_predict(net_448,net_512, tokenizer, valid_loader)
            log.write('time %s \n' % time_to_str(timer() - start_timer, 'min'))

            #np.save(submit_dir + '/probability.uint8.npy',probability)
            #write_pickle_to_file(submit_dir + '/predict.pickle', predict)
            #exit(0)
        else:
            pass
            #predict = read_pickle_from_file(submit_dir + '/predict.pickle')

        #----
        if is_norm_ichi:
            predict = [normalize_inchi(t) for t in predict]  #

        df_submit = pd.DataFrame()
        df_submit.loc[:,'image_id'] = df_valid.image_id.values
        df_submit.loc[:,'InChI'] = predict #
        df_submit.to_csv(submit_dir + '/submit.csv', index=False)

        log.write('submit_dir : %s\n' % (submit_dir))
        log.write('initial_checkpoint : %s\n' % (initial_checkpoint))
        log.write('df_submit : %s\n' % str(df_submit.shape))
        log.write('%s\n' % str(df_submit))
        log.write('\n')

        if 'local' in mode:
            truth = df_valid['InChI'].values.tolist()
            lb_score = compute_lb_score(predict, truth)
            #print(lb_score)
            log.write('lb_score  = %f\n'%lb_score.mean())
            log.write('is_norm_ichi = %s\n' % is_norm_ichi)
            log.write('\n')

            if 1:
                df_eval = df_submit.copy()
                df_eval.loc[:,'truth']=truth
                df_eval.loc[:,'lb_score']=lb_score
                df_eval.loc[:,'length'] = df_valid['length']
                df_eval.to_csv(submit_dir + '/df_eval.csv', index=False)
                 # df_valid.to_csv(submit_dir + '/df_valid', index=False)


        exit(0)


def cat_submit():
    df =[]
    for i in [3,2,1,0]:
        file=\
            '/root/share1/kaggle/2021/bms-moleular-translation/result/try10/tnt-s-224-fairse/fold3/valid/remote-00266000_model-gpu%d/submit.csv'%i
        d = pd.read_csv(file)
        df.append(d)

    df = pd.concat(df)
    print(df)
    print(df.shape)
    df.to_csv('/root/share1/kaggle/2021/bms-moleular-translation/result/try10/tnt-s-224-fairse/fold3/valid/submission-00266000_model.csv',index=False)




# main #################################################################
if __name__ == '__main__':
    run_submit()

''' 
40000 / 40000   7 min 58 sec
[40000 rows x 2 columns]
lb_score  = 1.964150


00266000_model
[10000 rows x 2 columns]
lb_score  = 1.953700
is_norm_ichi = False



00298000_model
[10000 rows x 2 columns] 
lb_score  = 1.890700
is_norm_ichi = False
'''