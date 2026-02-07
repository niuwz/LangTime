import os
import torch
from models.LangTime import LTModel, LTPratrainedModel, LTPredictModel
from configs.log_config import get_logger
logger = get_logger()

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.use_ds = os.path.isfile(args.deepspeed_config)
        self.model_dict = {
            "lt": LTPratrainedModel,
            "lt_rl": LTPredictModel,
            "lt_eval": LTPredictModel,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu and not self.use_ds:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            logger.info('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.use_ds:
            device_id = torch.cuda.current_device()
            device = torch.device('cuda:{}'.format(device_id))
            logger.info("local_rank: {}, use device cuda:{}".format(self.args.local_rank, device_id))
        else:
            device = torch.device('cpu')
            logger.info('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
