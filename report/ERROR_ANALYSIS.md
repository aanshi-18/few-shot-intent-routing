# Qualitative error audit

42 model-message cases (42 distinct test messages), split 0, sample 0, five-shot post-adaptation.

Six cases sampled per model for original misroutes, new misroutes and unknown acceptance; three supported rejections per model. Sampling seeds 42 and 17. Selected failure cases are not representative prevalence estimates. Notes are AI-assisted interpretations, not independently adjudicated ground truth. No labels or thresholds were changed.

## Case 1: TF-IDF + LR / original_misroute
Message: I transferred some money but it is yet to arrive.
Dataset label: `balance_not_updated_after_bank_transfer`; predicted: `transfer_not_received_by_recipient`.
Missing destination/account context; arrival wording is compatible with multiple transfer intents.

## Case 2: TF-IDF + LR / original_misroute
Message: How can someone add money to my account?
Dataset label: `topping_up_by_card`; predicted: `supported_cards_and_currencies`.
Vague add-money request lacks a specific funding method; lexical match is not sufficient.

## Case 3: TF-IDF + LR / original_misroute
Message: What can I use to verify my identify?
Dataset label: `verify_my_identity`; predicted: `why_verify_identity`.
Method of identity verification confused with reason for verification.

## Case 4: TF-IDF + LR / original_misroute
Message: Topping up my card is not working because I believe the top up is pending
Dataset label: `pending_top_up`; predicted: `top_up_failed`.
Pending and failed top-ups share failure wording; pending state is explicit.

## Case 5: TF-IDF + LR / original_misroute
Message: My transaction was just declined when I was using my disposable virtual card. What can I do?
Dataset label: `virtual_card_not_working`; predicted: `get_disposable_virtual_card`.
Virtual-card failure confused with obtaining a disposable card.

## Case 6: TF-IDF + LR / original_misroute
Message: What are the fees for top-ups?
Dataset label: `top_up_by_bank_transfer_charge`; predicted: `top_up_by_card_charge`.
Generic top-up fees lack the funding-method detail required by the label.

## Case 7: TF-IDF + LR / new_misroute
Message: I've tried numerous times to submit a transfer of funds. Why isn't it going through?
Dataset label: `beneficiary_not_allowed`; predicted: `transfer_not_received_by_recipient`.
Transfer failure description lacks explicit beneficiary restriction.

## Case 8: TF-IDF + LR / new_misroute
Message: I have tried to use my card several times and it does not work.
Dataset label: `card_not_working`; predicted: `declined_card_payment`.
Card-not-working and declined-payment intents overlap in wording.

## Case 9: TF-IDF + LR / new_misroute
Message: I want to transfer money to a beneficiery. Why can't I?
Dataset label: `beneficiary_not_allowed`; predicted: `topping_up_by_card`.
Beneficiary cue present, but sparse class fitting fails to select the new category.

## Case 10: TF-IDF + LR / new_misroute
Message: My exchange went wrong. I swapped Russian Ruble for UK pounds and was charged too much.
Dataset label: `card_payment_wrong_exchange_rate`; predicted: `wrong_exchange_rate_for_cash_withdrawal`.
Generic currency exchange complaint lacks payment versus withdrawal context.

## Case 11: TF-IDF + LR / new_misroute
Message: Why is my card not working?
Dataset label: `card_not_working`; predicted: `declined_card_payment`.
Short card failure request confused with the more represented declined-payment intent.

## Case 12: TF-IDF + LR / new_misroute
Message: I would like to know why I was charged twice for my purchase.
Dataset label: `transaction_charged_twice`; predicted: `request_refund`.
Duplicate charge is mistaken for a request to refund it; symptom versus remedy.

## Case 13: TF-IDF + LR / unknown_accepted
Message: Can you help me figure out what's happening? I'm trying to transfer money to a friend but it keeps getting returned. I'm not sure what i'm doing wrong.
Dataset label: `failed_transfer`; predicted: `transfer_not_received_by_recipient`.
Unknown failed transfer is close to a supported recipient-not-received intent.

## Case 14: TF-IDF + LR / unknown_accepted
Message: Do charge fees for transfers?
Dataset label: `transfer_fee_charged`; predicted: `top_up_by_bank_transfer_charge`.
Transfer-fee question is accepted as a supported bank-transfer top-up fee.

## Case 15: TF-IDF + LR / unknown_accepted
Message: How many days does it take until funds are in my account?
Dataset label: `transfer_timing`; predicted: `balance_not_updated_after_bank_transfer`.
Unknown transfer timing resembles delayed account-balance updating.

## Case 16: TF-IDF + LR / unknown_accepted
Message: Do you accept checks?
Dataset label: `top_up_by_cash_or_cheque`; predicted: `supported_cards_and_currencies`.
Unsupported cheque funding confused with supported payment instruments.

## Case 17: TF-IDF + LR / unknown_accepted
Message: Why is my cash withdrawal from the ATM still not yet showing as confirmed in my account?
Dataset label: `pending_cash_withdrawal`; predicted: `declined_cash_withdrawal`.
Pending withdrawal is semantically adjacent to supported declined withdrawal.

## Case 18: TF-IDF + LR / unknown_accepted
Message: I still don't have my card after 2 weeks. What should I do?
Dataset label: `card_arrival`; predicted: `lost_or_stolen_phone`.
Card-arrival request assigned an unrelated supported phone-loss label.

## Case 19: MiniLM prototypes / original_misroute
Message: I would like a temporary virtual card
Dataset label: `get_disposable_virtual_card`; predicted: `getting_virtual_card`.
Disposable versus general virtual-card distinction is not maintained.

## Case 20: MiniLM prototypes / original_misroute
Message: There must be an issue, why has my card been cancelled?
Dataset label: `reverted_card_payment?`; predicted: `declined_card_payment`.
Cancelled-card wording is ambiguous relative to the payment-reversion label.

## Case 21: MiniLM prototypes / original_misroute
Message: I withdrew some cash out of the ATM over the holiday. It seems that I was charged some outrageous fees. I would not have done that had I been aware of these outrageous charges!
Dataset label: `wrong_exchange_rate_for_cash_withdrawal`; predicted: `cash_withdrawal_charge`.
Text emphasizes fees rather than exchange rate; possible contextual ambiguity.

## Case 22: MiniLM prototypes / original_misroute
Message: Why can't I see my topup in my wallet anymore?
Dataset label: `topping_up_by_card`; predicted: `top_up_reverted`.
Missing top-up funds interpreted as reversion rather than how to top up.

## Case 23: MiniLM prototypes / original_misroute
Message: I'm really stuck. I don't know why but my card payment has not gone through.
Dataset label: `declined_card_payment`; predicted: `reverted_card_payment?`.
Payment not going through can describe decline or reversion without extra context.

## Case 24: MiniLM prototypes / original_misroute
Message: I have 1 other US card. Can you take that?
Dataset label: `supported_cards_and_currencies`; predicted: `getting_spare_card`.
Card acceptance interpreted as obtaining an additional card.

## Case 25: MiniLM prototypes / new_misroute
Message: I would like a refund on the extra pound I was charged.
Dataset label: `extra_charge_on_statement`; predicted: `request_refund`.
Refund wording dominates the small extra-charge intent.

## Case 26: MiniLM prototypes / new_misroute
Message: Why did I get a message saying that my transfer is not possible? I have done this before no problem. Please fix.
Dataset label: `beneficiary_not_allowed`; predicted: `declined_transfer`.
Beneficiary restriction and declined transfer are semantically close.

## Case 27: MiniLM prototypes / new_misroute
Message: Are there limits on auto top-up?
Dataset label: `automatic_top_up`; predicted: `top_up_limits`.
Request combines automatic top-up with a limits question; boundary overlap.

## Case 28: MiniLM prototypes / new_misroute
Message: Why am I being charged more ?
Dataset label: `card_payment_wrong_exchange_rate`; predicted: `extra_charge_on_statement`.
Very short extra-charge wording lacks the exchange-rate detail in its label.

## Case 29: MiniLM prototypes / new_misroute
Message: How come the ATM gave me less cash than what I asked for?
Dataset label: `wrong_amount_of_cash_received`; predicted: `declined_cash_withdrawal`.
Wrong cash amount is mistaken for declined withdrawal despite explicit less-cash cue.

## Case 30: MiniLM prototypes / new_misroute
Message: Why did I get a fee?
Dataset label: `cash_withdrawal_charge`; predicted: `extra_charge_on_statement`.
Generic fee request lacks withdrawal context.

## Case 31: MiniLM prototypes / unknown_accepted
Message: Do you guys accept Visa or Mastercard?
Dataset label: `visa_or_mastercard`; predicted: `supported_cards_and_currencies`.
Unknown card-network category overlaps a supported cards-and-currencies intent.

## Case 32: MiniLM prototypes / unknown_accepted
Message: Can I get a real card?
Dataset label: `order_physical_card`; predicted: `getting_spare_card`.
Unknown physical-card ordering resembles obtaining a spare card.

## Case 33: MiniLM prototypes / unknown_accepted
Message: my atm cash out is still pending
Dataset label: `pending_cash_withdrawal`; predicted: `declined_cash_withdrawal`.
Unknown pending withdrawal is accepted as declined withdrawal.

## Case 34: MiniLM prototypes / unknown_accepted
Message: Why was I not able to complete a transfer?
Dataset label: `failed_transfer`; predicted: `declined_transfer`.
Unknown failed transfer maps naturally to a supported declined-transfer category.

## Case 35: MiniLM prototypes / unknown_accepted
Message: im not paying this transfer fee
Dataset label: `transfer_fee_charged`; predicted: `top_up_by_bank_transfer_charge`.
Unknown transfer fee overlaps supported top-up-by-bank-transfer fee.

## Case 36: MiniLM prototypes / unknown_accepted
Message: How long does it take for a transfer?
Dataset label: `transfer_timing`; predicted: `pending_transfer`.
Unknown transfer timing overlaps supported pending-transfer wording.

## Case 37: TF-IDF + LR / supported_rejected
Message: i want to close an account but im not sure about setting up a new one in the future, what do you recommend
Dataset label: `terminate_account`; predicted: `age_limit`.
Long mixed request: closing account and reopening uncertainty; rejected wrong route.

## Case 38: TF-IDF + LR / supported_rejected
Message: Why are there so many fees on my statement?
Dataset label: `extra_charge_on_statement`; predicted: `extra_charge_on_statement`.
Correct top-one extra-charge classification is rejected by the global threshold.

## Case 39: TF-IDF + LR / supported_rejected
Message: where is theft-top option?
Dataset label: `automatic_top_up`; predicted: `verify_source_of_funds`.
Noisy theft-top phrase lacks clear automatic-top-up wording; rejected wrong route.

## Case 40: MiniLM prototypes / supported_rejected
Message: Something went wrong, I was charged an extra pound?
Dataset label: `extra_charge_on_statement`; predicted: `extra_charge_on_statement`.
Correct top-one extra-charge classification is rejected despite a relevant message.

## Case 41: MiniLM prototypes / supported_rejected
Message: I don't have my access code for the app.
Dataset label: `passcode_forgotten`; predicted: `passcode_forgotten`.
Correct forgotten-passcode classification is rejected; paraphrase uses access code.

## Case 42: MiniLM prototypes / supported_rejected
Message: I think my child used my card while I wasn't home.
Dataset label: `compromised_card`; predicted: `compromised_card`.
Correct compromised-card classification is rejected; child-use scenario differs from typical fraud wording.
