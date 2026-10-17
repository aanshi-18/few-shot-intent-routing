"""Stage 6: generate figures and report from saved results; no fitting or tuning."""
from pathlib import Path
import json
import html
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from matplotlib.font_manager import findfont, FontProperties
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("DejaVu", findfont(FontProperties(family="DejaVu Sans"))))
pdfmetrics.registerFont(TTFont("DejaVuBold", findfont(FontProperties(family="DejaVu Sans",weight="bold"))))
from reportlab.lib.pagesizes import A4

P=Path(__file__).resolve().parent
R=P/'report';R.mkdir(exist_ok=True)
D=pd.read_csv(P/'results/full_study/summary.csv')
labels={'tfidf_lr':'TF-IDF + LR','minilm_proto':'MiniLM prototypes'}
palette={'tfidf_lr':'#526D82','minilm_proto':'#167D8D'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def fig(name,cols,titles,percent):
    f,axes=plt.subplots(1,2,figsize=(9.2,3.4),constrained_layout=True)
    for ax,col,title in zip(axes,cols,titles):
        scale=100 if percent else 1
        for method in labels:
            x=D[D.method==method].sort_values('shots')
            ax.errorbar(x.shots,x[col+'__mean']*scale,yerr=x[col+'__split_sd']*scale,
                fmt='o-',capsize=4,color=palette[method],label=labels[method],linewidth=1.7)
        ax.set_xticks([1,5,10]);ax.set_xlabel('Fitting examples per new intent')
        ax.set_title(title,fontsize=11);ax.set_ylim(0,100 if percent else 1)
        ax.set_ylabel('Percent' if percent else 'Macro-F1');ax.grid(axis='y',alpha=.2)
    axes[0].legend(loc='lower right',fontsize=9)
    f.savefig(R/name,dpi=200);plt.close(f)
f,axes=plt.subplots(1,2,figsize=(9.2,3.4),constrained_layout=True)
for ax,col,title in zip(axes,['post__new_message_subset_accuracy','post__all_supported_closed_set_macro_f1'],['New-intent accuracy','All-supported macro-F1']):
    for method in labels:
        x=D[D.method==method].sort_values('shots');scale=100 if 'accuracy' in col else 1
        ax.errorbar(x.shots,x[col+'__mean']*scale,yerr=x[col+'__split_sd']*scale,fmt='o-',capsize=4,color=palette[method],label=labels[method])
    ax.set_xticks([1,5,10]);ax.set_xlabel('Fitting examples per new intent');ax.set_title(title)
    ax.set_ylim(0,100 if 'accuracy' in col else 1);ax.set_ylabel('Percent correct' if 'accuracy' in col else 'Macro-F1');ax.grid(axis='y',alpha=.2)
axes[0].legend(loc='lower right',fontsize=9)
f.savefig(R/'adaptation.png',dpi=200);plt.close(f)
fig('routing.png',['post__all_supported_correct_routing_rate','post__test_unknown_false_acceptance'],
    ['Supported requests correctly routed','Unknown requests incorrectly accepted'],True)

# Deterministic sample selection; this reads stored predictions, not model outputs rerun.
selected=[]
for method in ['tfidf_lr','minilm_proto']:
    data=pd.read_csv(P/'results/full_study/runs'/method/'split0/post_a0_k5/predictions.csv')
    for error_type,population in [('original_misroute','orig_supported'),('new_misroute','new'),('unknown_accepted','test_unknown')]:
        eligible=data[(data.population==population)&data.accepted]
        if error_type!='unknown_accepted':eligible=eligible[eligible.true!=eligible.pred]
        for _,row in eligible.sample(n=6,random_state=42).iterrows():
            selected.append(dict(method=method,error_type=error_type,row=int(row.row),text=row.text,true=row.true,pred=row.pred,score=row.score))
for method in ['tfidf_lr','minilm_proto']:
    data=pd.read_csv(P/'results/full_study/runs'/method/'split0/post_a0_k5/predictions.csv')
    eligible=data[data.population.isin(['orig_supported','new'])&~data.accepted]
    for _,row in eligible.sample(n=3,random_state=17).iterrows():
        selected.append(dict(method=method,error_type='supported_rejected',row=int(row.row),text=row.text,true=row.true,pred=row.pred,score=row.score))
sample=pd.DataFrame(selected);sample.insert(0,'case_id',range(1,len(sample)+1))
if (R/'error_sample.csv').exists():
    old=pd.read_csv(R/'error_sample.csv')
    assert old[['method','row']].equals(sample[['method','row']]),'Selection changed; review notes before regenerating.'
sample.to_csv(R/'error_sample.csv',index=False)

notes=[
'Missing destination/account context; arrival wording is compatible with multiple transfer intents.',
'Vague add-money request lacks a specific funding method; lexical match is not sufficient.',
'Method of identity verification confused with reason for verification.',
'Pending and failed top-ups share failure wording; pending state is explicit.',
'Virtual-card failure confused with obtaining a disposable card.',
'Generic top-up fees lack the funding-method detail required by the label.',
'Transfer failure description lacks explicit beneficiary restriction.',
'Card-not-working and declined-payment intents overlap in wording.',
'Beneficiary cue present, but sparse class fitting fails to select the new category.',
'Generic currency exchange complaint lacks payment versus withdrawal context.',
'Short card failure request confused with the more represented declined-payment intent.',
'Duplicate charge is mistaken for a request to refund it; symptom versus remedy.',
'Unknown failed transfer is close to a supported recipient-not-received intent.',
'Transfer-fee question is accepted as a supported bank-transfer top-up fee.',
'Unknown transfer timing resembles delayed account-balance updating.',
'Unsupported cheque funding confused with supported payment instruments.',
'Pending withdrawal is semantically adjacent to supported declined withdrawal.',
'Card-arrival request assigned an unrelated supported phone-loss label.',
'Disposable versus general virtual-card distinction is not maintained.',
'Cancelled-card wording is ambiguous relative to the payment-reversion label.',
'Text emphasizes fees rather than exchange rate; possible contextual ambiguity.',
'Missing top-up funds interpreted as reversion rather than how to top up.',
'Payment not going through can describe decline or reversion without extra context.',
'Card acceptance interpreted as obtaining an additional card.',
'Refund wording dominates the small extra-charge intent.',
'Beneficiary restriction and declined transfer are semantically close.',
'Request combines automatic top-up with a limits question; boundary overlap.',
'Very short extra-charge wording lacks the exchange-rate detail in its label.',
'Wrong cash amount is mistaken for declined withdrawal despite explicit less-cash cue.',
'Generic fee request lacks withdrawal context.',
'Unknown card-network category overlaps a supported cards-and-currencies intent.',
'Unknown physical-card ordering resembles obtaining a spare card.',
'Unknown pending withdrawal is accepted as declined withdrawal.',
'Unknown failed transfer maps naturally to a supported declined-transfer category.',
'Unknown transfer fee overlaps supported top-up-by-bank-transfer fee.',
'Unknown transfer timing overlaps supported pending-transfer wording.',
'Long mixed request: closing account and reopening uncertainty; rejected wrong route.',
'Correct top-one extra-charge classification is rejected by the global threshold.',
'Noisy theft-top phrase lacks clear automatic-top-up wording; rejected wrong route.',
'Correct top-one extra-charge classification is rejected despite a relevant message.',
'Correct forgotten-passcode classification is rejected; paraphrase uses access code.',
'Correct compromised-card classification is rejected; child-use scenario differs from typical fraud wording.'
]
audit=pd.read_csv(R/'error_sample.csv');assert len(audit)==len(notes)==42
audit['review_note']=notes;audit['reviewer']='AI-assisted qualitative audit'
audit.to_csv(R/'error_audit.csv',index=False)
assert not audit.duplicated(['method','row']).any()
unique=audit.row.nunique()
details=['# Qualitative error audit','',f'42 model-message cases ({unique} distinct test messages), split 0, sample 0, five-shot post-adaptation.','',
 'Six cases sampled per model for original misroutes, new misroutes and unknown acceptance; three supported rejections per model. Sampling seeds 42 and 17. Selected failure cases are not representative prevalence estimates. Notes are AI-assisted interpretations, not independently adjudicated ground truth. No labels or thresholds were changed.','']
for _,r in audit.iterrows():
    details += [f'## Case {r.case_id}: {labels[r.method]} / {r.error_type}',f'Message: {r.text}',
                f'Dataset label: `{r.true}`; predicted: `{r.pred}`.',str(r.review_note),'']
(R/'ERROR_ANALYSIS.md').write_text('\n'.join(details))

def value(method,k,key):return float(D[(D.method==method)&(D.shots==k)].iloc[0][key+'__mean'])
newgain=100*(value('minilm_proto',5,'post__new_message_subset_accuracy')-value('tfidf_lr',5,'post__new_message_subset_accuracy'))
tables=[]
for method in ['tfidf_lr','minilm_proto']:
 for k in [1,5,10]:
  row=D[(D.method==method)&(D.shots==k)].iloc[0]
  tables.append([labels[method],str(k),f"{100*row['post__new_message_subset_accuracy__mean']:.2f}",f"{row['post__all_supported_closed_set_macro_f1__mean']:.3f}",f"{100*row['post__all_supported_correct_routing_rate__mean']:.2f}",f"{100*row['post__test_unknown_false_acceptance__mean']:.2f}"])

pages=[
 [('title','Few-Shot Customer Support Routing with Unknown-Intent Detection'),
  ('p','Aanshi Arora | IIT Kharagpur | Undergraduate benchmarking study | October 2026'),
  ('h','Abstract'),
  ('p',f'This study compares TF-IDF/logistic regression with frozen MiniLM sentence embeddings and category prototypes when a banking-support system expands its intent inventory. BANKING77 categories are divided into 45 initially supported, 15 newly introduced, seven development-only unknown and ten final-test unknown intents. New categories receive 1, 5 or 10 fitting examples, with additional labelled development data for rejection-threshold calibration. Across three category splits and three adaptation samples, prototypes achieved mean new-intent accuracies of 41.94%, 75.70% and 79.91%, compared with 0.35%, 11.93% and 30.78% for unweighted logistic regression. At five shots, the gap is {newgain:.2f} percentage points. However, unknown false acceptance remained high for both methods. The results establish an adaptation advantage under this protocol, not a reliable production rejection system or a causal comparison of equally resourced representations.'),
  ('h','1. Motivation and related work'),
  ('p','Customer-support routing must distinguish between supported requests and unfamiliar needs. Adding a service changes the supported inventory while leaving other requests out of scope. Classification accuracy alone can obscure two costly failures: routing unfamiliar requests automatically and escalating requests that could have been handled correctly.'),
  ('p','Casanueva et al. [1] introduced BANKING77 and investigated pretrained sentence encoders for low-data intent detection. Larson et al. [2] showed the importance of evaluating out-of-scope prediction separately from supported-intent classification. Snell et al. [3] provide the prototype-based few-shot learning principle. Here, prototypes are built on a frozen existing encoder; this is not a reproduction of their learned episodic representation or a proposed new algorithm.'),
  ('h','Research questions'),
  ('p','RQ1. How does new-intent classification change with 1, 5 and 10 fitting examples?'),
  ('p','RQ2. How does adding categories affect correct routing and rejection of still-unknown requests?'),
  ('p','RQ3. What failure patterns are visible in a targeted qualitative audit?'),
  ('p','The contribution is a reproducible controlled benchmark with matched samples, explicit calibration budgets, checked prediction provenance and analysis of adaptation versus rejection.')],
 [('h','2. Dataset and experimental protocol'),
  ('p','BANKING77 contains 10,003 training and 3,080 test messages across 77 labels; each official test category has 40 messages. Raw data and its CC BY 4.0 licence are retained. Lowercasing and whitespace collapse identify seven training rows that also occur in the official test set. These rows are excluded before any learning or development partition. Official test rows remain unchanged. Model input collapses whitespace; the encoder retains case, while TF-IDF lowercases.'),
  ('p','Category-split seeds 0, 1 and 2 allocate 45 original, 15 new, seven development-unknown and ten test-unknown labels. Approximately 20% of duplicate-key groups in original/new training categories become development messages. Remaining original messages fit the original classifier. New-intent adaptation seeds 0, 1 and 2 draw distinct messages with nested prefixes of 1, 5 and 10 examples. Other new-intent training messages are unused. Final-test unknown categories never enter fitting or calibration.'),
  ('p','Before addition, both new and final-test unknown intents are unsupported and reported separately. After addition, 60 intents are supported. Each evaluation has 1,800 original, 600 new and 400 final-test unknown messages. Post-adaptation development sets contain 1,546, 1,554 and 1,573 messages across the three splits, including 443, 390 and 349 labelled new-intent messages. Thus k-shot describes fitting data, not the complete labelled-data budget.'),
  ('h','3. Models and rejection'),
  ('p','TF-IDF uses word unigrams/bigrams, sublinear term frequency and min_df=1. Multiclass logistic regression uses C=10, lbfgs and max_iter=1000 without class weighting. On addition, vocabulary and classifier are refitted on original training messages plus new examples. Settings are fixed; they are not tuned against final test outcomes.'),
  ('p','MiniLM uses sentence-transformers/all-MiniLM-L6-v2, revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Independent frozen inference creates 384-dimensional L2-normalized vectors. A category prototype is the mean of its fitting vectors, normalized again. Prediction selects the largest cosine similarity. Adding categories preserves original prototypes. No encoder training is performed; no messages were truncated at the 256-token limit.'),
  ('p','The acceptance score is maximum LR probability or maximum prototype cosine. The threshold is the lower 10th percentile of supported-development scores; accept if score >= threshold. Before addition, calibration uses original dev messages. After addition, it uses original plus new dev messages. Development unknown outcomes are diagnostics only. Neither score is assumed to be calibrated confidence.'),
  ('h','Metrics and aggregation'),
  ('p','New accuracy ignores rejection; correct routing counts only accepted correct predictions. Unknown false acceptance is the accepted fraction of final-test unknown messages. Per-category F1 uses all currently supported messages, including cross-group false positives. Each method has 27 post configurations and three pre baselines. Sampling seeds are averaged within each category split; three split means are equally weighted. Error bars show descriptive SD across split means, not independent-dataset confidence intervals.')],
 [('h','4. Adaptation results'),
  ('table',tables),
  ('p','Table 1. Means across the repeated design. New accuracy, correct routing and unknown false acceptance are percentages. F1 is on the 0-1 scale. Additional labelled development data is used in every post-adaptation setting.'),
  ('image','adaptation.png'),
  ('p','Figure 1. New-intent accuracy and all-supported macro-F1. Error bars are SD across three category-split means after averaging adaptation samples within each split. The overlapping splits are not independent datasets.'),
  ('p','Prototype accuracy rises substantially from one to five fitting examples, then more modestly from five to ten. The unweighted LR baseline also improves with additional examples but remains much weaker on new intents. At ten shots, prototypes reach 79.91% mean new accuracy and 0.856 all-supported F1; LR reaches 30.78% and 0.731.'),
  ('p','This contrast is compatible with a benefit from pretrained semantic representations, but it does not isolate that cause. LR faces severe imbalance between many original examples and few new ones, and the prototype classifier allocates one representative vector per category. Class balancing and a classifier trained on the same embeddings would be useful future ablations. They were not executed in this study.'),
  ('p','The comparison therefore concerns these two practical pipelines under matched adaptation samples and fixed configurations. It does not establish that all linear models fail at few-shot learning, or that prototypes universally outperform fine-tuning.')],
 [('h','5. Routing reliability and qualitative findings'),
  ('image','routing.png'),
  ('p','Figure 2. Accepted correct routing on supported messages versus false acceptance of unknown messages. Lower is better only in the right panel. These are different populations and should not be combined without an explicit deployment mixture or cost model.'),
  ('p','At five shots, prototypes correctly route 78.88% of supported requests versus 69.82% for LR. Yet unknown false acceptance is 69.25% versus 69.31%. At ten shots, LR has slightly lower unknown false acceptance (66.50% versus 67.86%) despite weaker adaptation. Neither model provides reliable unknown rejection at the chosen high-coverage operating point.'),
  ('p','Supported test acceptance stays near the 90% development target: approximately 89.9-90.8% for LR and 89.5-90.1% for prototypes across shot counts. This does not imply an unknown-error guarantee. Comparisons are at a shared calibration target, not exactly matched test coverage. Selecting a stricter threshold based on test results would invalidate the primary evaluation.'),
  ('h','AI-assisted qualitative audit'),
  ('p',f'Forty-two model-message cases ({unique} distinct messages) were inspected from split 0, adaptation seed 0, five-shot outputs. Per model, six cases were sampled for original misroutes, new misroutes and accepted unknowns, plus three supported rejections. Sampling seeds were fixed at 42 and 17. This selected sample illustrates failure mechanisms; its category proportions do not estimate their prevalence. Interpretations are AI-assisted and not independently adjudicated.'),
  ('p','Semantically adjacent labels create difficult boundaries. For example, an unknown failed-transfer request was accepted as declined_transfer by prototypes; an unknown transfer-fee request was accepted as a supported bank-transfer top-up fee. A prototype prediction for "Are there limits on auto top-up?" selected top_up_limits rather than automatic_top_up. These messages can combine multiple cues even when the benchmark assigns one intent.'),
  ('p','Some rejections discard correct top-one predictions. Prototype cases about a forgotten access code and an extra pound charged were correctly classified but rejected by the global threshold. Conversely, LR routed a card-arrival request to lost_or_stolen_phone. The audit suggests both fine-grained boundary ambiguity and score/rejection failures; it does not establish causal explanations or justify relabelling data. All 42 case notes are supplied separately.')],
 [('h','6. Limitations and reproducibility'),
  ('p','Calibration budget: labelled new-intent development messages exceed the 1/5/10 fitting examples. No strict k-shot end-to-end claim is made. Representation resources: MiniLM benefits from external pretraining; pretraining overlap with BANKING77 was not audited. Baseline imbalance: unweighted LR is a fixed baseline, not a tuned or class-balanced competitor.'),
  ('p','Generalization: category holdouts are artificial, the benchmark is English banking text, and only three overlapping category splits are used. No other domain or real deployment is evaluated. Rare labels, genuinely new services and real user traffic may behave differently. Rejection: the global threshold targets supported coverage rather than optimizing unknown-error cost. Both methods fail frequently on held-out unknowns.'),
  ('p','Environment: two verified post-adaptation pilot outputs and two pre phases were reused. The LR pilot records scikit-learn 1.9.1; newly executed Stage 5 LR runs use 1.8.0. Per-run provenance documents this mixed environment; it is not an exact single-environment replication. Prototype update timing excludes cached encoder inference, so no end-to-end latency superiority is claimed.'),
  ('p','All 60 saved phases were verified against split identities, file checksums, permitted labels, test rows and calibration scores. Core metrics were independently recomputed for all 54 post configurations. Each checkpoint retains predictions, development scores and metadata. Embeddings are cached with model revision, data hashes and preprocessing identity. Analysis introduces no new fitting or test-driven threshold changes.'),
  ('h','7. Conclusion'),
  ('p','Frozen MiniLM prototypes provide substantially stronger new-intent recognition than the fixed unweighted TF-IDF/LR pipeline under this benchmark protocol. That advantage does not solve open-set routing: unknown false acceptance remains high. The defensible outcome is a measured adaptation-versus-rejection trade-off and a reproducible baseline study. Future work should isolate class imbalance and representation effects and evaluate development-chosen rejection operating points under stricter label budgets.'),
  ('h','References'),
  ('p','[1] Casanueva, I., et al. (2020). Efficient Intent Detection with Dual Sentence Encoders. NLP4ConvAI, pp. 38-45. https://aclanthology.org/2020.nlp4convai-1.5/'),
  ('p','[2] Larson, S., et al. (2019). An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. EMNLP-IJCNLP. https://aclanthology.org/D19-1131/'),
  ('p','[3] Snell, J., Swersky, K., and Zemel, R. (2017). Prototypical Networks for Few-shot Learning. NeurIPS. https://arxiv.org/abs/1703.05175'),
  ('p','[4] Sentence Transformers. all-MiniLM-L6-v2 model card, pinned revision above. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2')]
]

# Verify report-specific protocol counts directly from the stored splits.
for seed,total,newcount in [(0,1546,443),(1,1554,390),(2,1573,349)]:
    d=json.loads((P/f'results/splits/split_seed{seed}.json').read_text())
    assert len(d['dev_new_rows'])==newcount and len(d['dev_supported_rows'])+newcount==total
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyReport',fontName='DejaVu',fontSize=9.5,leading=13,spaceAfter=8))
styles.add(ParagraphStyle(name='ReportTitle',fontName='DejaVuBold',fontSize=19,leading=23,spaceAfter=12,textColor=colors.HexColor('#173B50')))
styles.add(ParagraphStyle(name='ReportHead',fontName='DejaVuBold',fontSize=12,leading=15,spaceBefore=7,spaceAfter=7,textColor=colors.HexColor('#173B50')))
story=[];md=[]
for i,page in enumerate(pages):
 if i:story.append(PageBreak())
 for kind,content in page:
  if kind in ['title','h','p']:
   style={'title':'ReportTitle','h':'ReportHead','p':'BodyReport'}[kind]
   story.append(Paragraph(html.escape(content),styles[style]))
   md.append(('# ' if kind=='title' else '## ' if kind=='h' else '')+content+'\n')
  elif kind=='image':
   story.append(Image(str(R/content),width=455,height=168));story.append(Spacer(1,6))
   md.append(f'![Results]({content})\n')
  elif kind=='table':
   headings=['Model','Shots','New acc.','F1','Correct\nroute','Unknown\naccepted']
   table=Table([headings]+content,colWidths=[115,35,66,50,86,95])
   table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#173B50')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTNAME',(0,0),(-1,0),'DejaVuBold'),('FONTNAME',(0,1),(-1,-1),'DejaVu'),('FONTSIZE',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EFF4F6'),colors.white]),('ALIGN',(1,0),(-1,-1),'RIGHT')]))
   story.append(table);story.append(Spacer(1,9))
   md+=['| Model | Shots | New accuracy (%) | F1 | Correct routing (%) | Unknown accepted (%) |','|---|---:|---:|---:|---:|---:|']+['| '+' | '.join(row)+' |' for row in content]+['']
(R/'REPORT.md').write_text('\n'.join(md))
def footer(canvas,doc):
 canvas.setFont('DejaVu',7);canvas.setFillColor(colors.HexColor('#526575'))
 canvas.drawString(48,28,'BANKING77 | Few-shot fitting with additional calibration data')
 canvas.drawRightString(A4[0]-48,28,str(doc.page))
doc=SimpleDocTemplate(str(R/'Banking77_Research_Report.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=44,bottomMargin=45,title='Few-Shot Customer Support Routing with Unknown-Intent Detection',author='Aanshi Arora')
doc.build(story,onFirstPage=footer,onLaterPages=footer)

(P/'CV_BULLETS.md').write_text(f'''# Suggested CV entry

Few-Shot Intent Adaptation and Unknown-Request Detection | NLP Benchmarking Project

- Benchmarked TF-IDF/logistic regression against frozen MiniLM prototypes on BANKING77 across 54 post-adaptation configurations, using held-out intent categories and development-only rejection calibration.
- Achieved 75.70% mean new-intent accuracy with five fitting examples per category using prototypes, versus 11.93% for the fixed unweighted LR baseline; documented the additional labelled calibration-data budget.
- Audited duplicate leakage, independently verified prediction metrics, and analyzed persistent unknown false acceptance through a 42-case qualitative audit.

Do not claim an LLM was trained, a new algorithm was invented, or unknown detection was solved. The report and repository should accompany these bullets. The qualitative audit was AI-assisted; review the case notes yourself before discussing them as your own analysis.
''')
(P/'INTERVIEW_GUIDE.md').write_text('''# Understanding and defending this study

1. What is the application? Route supported banking requests, escalate unsupported ones, and add categories from a few examples.
2. What is an intent? A label for the user's support need. Some benchmark boundaries are semantically close.
3. What did you train? TF-IDF vocabulary and logistic regression. MiniLM was frozen; prototypes were averaged fitting embeddings.
4. Why only a few examples? New categories may lack annotations. However, this study also uses additional new-category development labels; it is not strict k-shot end-to-end.
5. Why hold out categories? Holding out messages tests familiar-category generalization; holding out categories tests requests the inventory does not support.
6. How are prototypes built? L2-normalize embeddings, average category fitting vectors, normalize the average, compare cosine similarity. Original prototypes do not change on category addition.
7. How does rejection work? Maximum probability/cosine is compared with a development-only threshold. The lower 10th percentile targets 90% supported acceptance, not 90% correctness or a guaranteed unknown-error rate.
8. Why is LR weak on new intents? Class imbalance and lack of pretrained semantic representations are plausible contributors. We did not isolate them; do not present a causal conclusion.
9. What did the results establish? Better adaptation for this prototype pipeline under this protocol, while unknown rejection remains poor for both.
10. Is higher confidence reliable? Not automatically. Cosine similarity and LR maximum probability are different scores, not directly comparable calibrated confidences.
11. What does the F1 metric count? Per-category F1 counts errors across all supported messages, including old/new cross-group false positives. Subset F1 is a separate diagnostic.
12. Why not claim statistical significance? Three overlapping category splits do not constitute independent datasets. We report descriptive split variation.
13. What remains limited? Single domain, artificial inventory holdouts, extra calibration labels, encoder pretraining overlap untested, mixed pilot versions and a fixed rejection operating point.
14. What would you test next? Class-balanced LR, a classifier using the same embeddings, and development-selected stricter rejection targets. These are proposed follow-ups, not completed results.
15. What must you personally review? Read the source functions, trace one example through prediction and thresholding, review the 42 audit cases, and explain the counts and denominators in each metric.

Resources: scikit-learn text analytics and classification metrics; Sentence Transformers quickstart/model card; Casanueva et al. 2020; Larson et al. 2019; Snell et al. 2017. URLs are in the report and README. Repository code was AI-assisted: learn, check and own the reasoning rather than claiming unaided implementation.
''')
print('Generated report, two figures, 42-case audit, CV bullets and interview guide.')
