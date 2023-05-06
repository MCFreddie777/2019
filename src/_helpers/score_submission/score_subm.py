from . import functions as f


def main(df_subm, df_gt):
    mrr, map3 = f.score_submissions(df_subm, df_gt)
    
    print(f'Mean reciprocal rank:      {mrr}')
    print(f'Mean average precision @3: {map3}')
    
    return mrr, map3
