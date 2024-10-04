import math
from typing import Dict, List, Optional
import sys

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from fairseq.ngram_repeat_block import NGramRepeatBlock

from data import data_utils



class Custom_Model(nn.Module):
    def __init__(
        self,
        args,
        model,
        tgt_dict,
        max_len=0,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
        constraint_trie=None,
        constraint_range=None,
        gen_code=False,
        gen_box=False,
        ignore_eos=False,
        zero_shot=False,
        get_encoder_att = False
    ): 
        # input: constraint_range = cfg.constraint_range,
        super().__init__()
        self.model = model
        
        # get sequence generator parameters
        
        search_strategy = search.BeamSearch(tgt_dict)
        beam_size=getattr(args, "beam", 5)
        max_len_a=getattr(args, "max_len_a", 0)
        max_len_b=getattr(args, "max_len_b", 200)
        min_len=getattr(args, "min_len", 1)
        normalize_scores=(not getattr(args, "unnormalized", False))
        len_penalty=getattr(args, "lenpen", 1)
        unk_penalty=getattr(args, "unkpen", 0)
        temperature=getattr(args, "temperature", 1.0)
        match_source_len=getattr(args, "match_source_len", False)
        no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0)
        
        self.gen_code = gen_code
        self.gen_box = gen_box
        self.ignore_eos = ignore_eos
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.bos = tgt_dict.bos()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.max_len = max_len or self.model.max_decoder_positions()

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.zero_shot = zero_shot

        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

        self.constraint_trie = constraint_trie

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)
        self.get_encoder_att = get_encoder_att

    def cuda(self):
        self.model.cuda()
        return self

    def forward(self, net_input):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
            ],
        )
        src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (
            (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        )
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = int(self.max_len_a * src_len + self.max_len_b)
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        encoder_out = self.model.encoder.forward(**encoder_input, get_encoder_att=False)
        encoder_att_weights = [encoder_out['encoder_att_weights']]
        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_out = self.model.encoder.reorder_encoder_out(encoder_out, new_order)
        
        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()   # score is the lprob accumulated across steps
        )  # +1 for eos; pad is never chosen for scoring
        attn: Optional[Tensor] = None
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        # tokens[:, 0] = self.eos if bos_token is None else bos_token
        tokens[:, 0] = self.bos
        attn: Optional[Tensor] = None 
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining
        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS
        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)
        
        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None
        original_batch_idxs = torch.arange(0, bsz).type_as(tokens)        
        
        for step in range(max_len + 1):  # one extra step for EOS marker
            decoder_input_tokens = tokens[:, : step + 1]
            code_mask = (tokens.new_ones(decoder_input_tokens.size(0))*self.gen_code).bool()
            decoder_out = self.model.decoder.forward(
                    decoder_input_tokens,
                    code_masks=code_mask,
                    encoder_out=encoder_out)
            avg_attn_scores = decoder_out[1]["attn"][0]
            avg_attn_scores = avg_attn_scores[:, -1, :]
            decoder_len = len(decoder_out)
            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(self.temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )
            lprobs = self.model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            lprobs = lprobs[:, -1, :]
            if step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf
            
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty
            
            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf
                if self.ignore_eos:
                    lprobs[:, self.eos] = 1
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)
            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)            
          # Shape: (batch, cand_size)   note: in step 0, cand_beams will always be 0 and only cand_indices is important
          # NOTE: cand_scores is the cumulative probability score from step 0 to current step
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )  
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask) 
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )        
            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)
            
            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break   
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"
            
            
            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once) comment: the tokens may be totally replaced for some samples
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        if not self.get_encoder_att:
            return finalized
        else:
            return finalized, encoder_att_weights 


    
    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = bbsz_idx // beam_size
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)  # this is for multi-sample batch, to recover the original index when the previous finished sample has been removed

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        sent_list: List[int] = sent.tolist()
        
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent_list[i]].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": eos_scores[i],
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                        "sum_scores": scores.index_select(0, bbsz_idx)[:, : step + 1]
                    }
                )

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False


class GradCAM_model(nn.Module):
    def __init__(self, model)