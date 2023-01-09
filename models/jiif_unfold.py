import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

@register('jiif-unfold')
class JIIF(nn.Module):
    def __init__(self, encoder_lr_spec, encoder_hr_spec, imnet_spec=None):
        super().__init__()

        self.encoder_lr = models.make(encoder_lr_spec)
        self.encoder_hr = models.make(encoder_hr_spec)

        imnet_in_dim = self.encoder_lr.out_dim 
        imnet_in_dim += self.encoder_hr.out_dim * 2
        imnet_in_dim *= 9 #feature unfolding
        imnet_in_dim += 2 #x,y coord
        #imnet_in_dim += 2 #cell decoding


        self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})

    def gen_feat(self, inp_lr, inp_hr):
        self.feat_lr = self.encoder_lr(inp_lr)
        self.feat_hr = self.encoder_hr(inp_hr)
        return self.feat_lr, self.feat_hr

    def query_ssh(self, coord, cell=None):
        feat_lr = self.feat_lr
        feat_hr = self.feat_hr

        #feature unfolding
        feat_lr = F.unfold(feat_lr, 3, padding=1).view(feat_lr.shape[0], feat_lr.shape[1]*9, feat_lr.shape[2], feat_lr.shape[3])
        feat_hr = F.unfold(feat_hr, 3, padding=1).view(feat_hr.shape[0], feat_hr.shape[1]*9, feat_hr.shape[2], feat_hr.shape[3])
        
        #local ensemble
        vx_lst = [-1,1]
        vy_lst = [-1,1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat_lr.shape[-2] / 2
        ry = 2 / feat_lr.shape[-1] / 2

        feat_coord_lr = make_coord(feat_lr.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat_lr.shape[0], 2, *feat_lr.shape[-2:])
        feat_coord_hr = make_coord(feat_hr.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat_hr.shape[0], 2, *feat_hr.shape[-2:])

        preds = []
        weights = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat_lr = F.grid_sample(
                    feat_lr, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord_lr, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_feat_hr = F.grid_sample(
                    feat_hr, q_coord.flip(-1).unsqueeze(1),
                    mode='bicubic', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                i_feat_hr = F.grid_sample(
                    feat_hr, coord_.flip(-1).unsqueeze(1),
                    mode='bicubic', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat_lr.shape[-2]
                rel_coord[:, :, 1] *= feat_lr.shape[-1]

                rel_feat = i_feat_hr - q_feat_hr
                rel_feat[:, :, 0] *= feat_lr.shape[-2]
                rel_feat[:, :, 1] *= feat_lr.shape[-1]
                inp = torch.cat([q_feat_lr, q_feat_hr, rel_feat, rel_coord], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat_lr.shape[-2]
                rel_cell[:, :, 1] *= feat_lr.shape[-1]
                #inp = torch.cat([inp, rel_cell], dim=-1)

                #print('input shape : ',inp.shape)
                #print('detail : ', q_feat_lr.shape, q_feat_hr.shape, rel_feat.shape, rel_coord.shape, rel_cell.shape)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                #print(pred.shape)

                if self.imnet.out_dim == 2 :
                    preds.append(pred[:,:,0].unsqueeze(-1))
                    weights.append(pred[:,:,1].unsqueeze(-1))

                elif self.imnet.out_dim == 1 :
                    preds.append(pred)

                    area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                    areas.append(area + 1e-9)

        if self.imnet.out_dim == 2 :
            #print(torch.stack(weights).shape)
            #print(torch.stack(preds).shape)

            #weights = torch.exp(torch.FloatTensor(weights))
            weights = torch.exp(torch.stack(weights))
            tot_weights = weights.sum(dim=0)

            #t = areas[0]; areas[0] = areas[3]; areas[3] = t
            #t = areas[1]; areas[1] = areas[2]; areas[2] = t
            ret = 0
            for pred, weight in zip(preds, weights):
                #print( pred.shape, weight.shape, tot_weights.shape)
                ret = ret + torch.mul(pred, torch.sub(weight, tot_weights))
            return ret
        
        elif self.imnet.out_dim == 1 :
            #print('on interpole')
            tot_area = torch.stack(areas).sum(dim=0)

            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
            ret = 0
            for pred, area in zip(preds, areas):
                #print(pred.shape, area.shape)
                #print('area : ', area[0][0])
                #print('tot area', tot_area)
                ret = ret + pred * (area / tot_area).unsqueeze(-1)
            #print('new batch')
            return ret

    def forward(self, inp_lr, inp_hr, coord, cell):
        self.gen_feat(inp_lr, inp_hr)
        return self.query_ssh(coord, cell)
