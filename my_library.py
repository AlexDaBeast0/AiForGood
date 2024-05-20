def test_load():
  return 'MAYA IS STINKY'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, A, ax, B, bx):
  subTable = up_table_subset(table, A, 'equals', ax)
  subList  = up_get_column(subTable, B)

  AColumn = up_get_column(table, A)
  BColumn = up_get_column(table, B)

  pBA = sum([1 if i == bx else 0 for i in subList])/len(subList)
  pA  = sum([1 if i == ax else 0 for i in AColumn])/len(AColumn)
  pB  = sum([1 if i == bx else 0 for i in BColumn])/len(BColumn)

  return pBA * pA / pB

def cond_probs_product(table, evidenceRow, target, targetVal):
  zipColumnRow = up_zip_lists(table.columns[:-1], evidenceRow)
  probList = []

  for i, j in zipColumnRow:
      probList += [cond_prob(table, i, j, target, targetVal)]

  return up_product(probList)

def naive_bayes(table, evidence_row, target):
  cond_prob_no = cond_probs_product(table, evidence_row, target, 0)
  prior_prob_no = prior_prob(table, target, 0)

  cond_prob_yes = cond_probs_product(table, evidence_row, target, 1)
  prior_prob_yes = prior_prob(table, target, 1)

  prob_target_no = cond_prob_no * prior_prob_no
  prob_target_yes = cond_prob_yes * prior_prob_yes
  
  neg, pos = compute_probs(prob_target_no, prob_target_yes)
  return [neg, pos]
