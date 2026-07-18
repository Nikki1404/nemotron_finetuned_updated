Hi Nikita,
 
Here's the background of our hackathon task:
 
Ghost Denials uses a counterfactual equivalence audit to make “silent” wrongful automated denials measurable and actionable. Starting with health-plan/insurance claim and prior-authorization auto-denials (excluding manual denials, fraud-flagged cases, and incomplete-submission rejections), it learns the decision-relevant structured feature set from historical adjudication logs (e.g., diagnosis/procedure codes, plan/benefit terms, clinical criteria, dollar bands, network status) and forms equivalence classes over those features. For each auto-denied case, it retrieves approved “near-twins” via propensity-score matching on the policy’s own decision function. If approved twins are statistically indistinguishable across all decision-relevant variables, the denial is flagged as a candidate wrongful denial and shipped with a minimal counterexample dossier (“this near-identical case was paid”). Risk of unobserved confounding is mitigated by requiring agreement on documentation-completeness and manual-review flags, excluding any case touched by manual clinical review, and validating the method against known ground truth. Ground truth comes from denials later overturned on appeal; the matcher is validated on these known outcomes before being applied to un-appealed denials. A conformal prediction layer provides calibrated confidence with a bounded false-positive rate (tunable to reviewer capacity). Disparate-impact tests run across protected attributes to surface bias, and matched data is also used to infer the decision boundary actually enforced and diff it against written benefit/medical policy to surface systemic rule divergences. The system is read-only/offline and produces a prioritized human-review queue; clinical/appeals staff approve any correction. An LLM is used only to draft plain-language review/appeal dossiers from structured evidence, not to render the wrongful-denial judgment.
Business Outcomes
ROI is proven immediately via a “back-book” scan of the last 12–24 months of auto-denials to surface high-confidence candidates, route them to clinical/appeals review, and correct them—then run continuously to detect new issues within days instead of only on appeal or never. The business case is supported by observed high overturn rates on appealed denials (e.g., Medicare Advantage prior-auth appeal overturns reported >80% in some CMS reporting); if even a fraction holds for the ~99% who never appeal, a mid-size plan could have tens of thousands of wrongful denials unmeasured today. Targets/KPIs: detect ≥80% of wrongful auto-denials at a bounded ≤10% false-positive rate (conformal-calibrated; tunable to reviewer capacity); reduce time-to-detection from “never/on-appeal” to days; quantify dollars and members recovered; and quantify/flag disparate-impact exposure. Measurement plan: offline precision/recall against appeal-overturn ground truth plus reviewer-confirmed precision on flagged queues; then track live wrongful-denial rate and overturn rate pre/post. Each “leak pattern class” (e.g., criteria mismatch, dollar-band edge cases, network-status errors) ships with acceptance tests and per-class error budgets for governed precision and regression testing. Commercial model: per-recovered-decision fee + continuous-assurance subscription + pattern-pack licensing.
 
I did a basic research and here are few things we can start with, unless you have any other idea:
 
dataset: Synthetic AR Medical Dataset with Realistic Denial
Algorithms:
Exact Matching
logistic regression or gradient boosting.
k-Nearest Neighbors
FAISS/embeddings
 
Synthetic AR Medical Billing Dataset with Realistic Denial Workflow

https://www.kaggle.com/datasets/abuthahir1998/synthetic-ar-medical-dataset-with-realistic-denial
