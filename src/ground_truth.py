import pandas as pd

from _helpers import constants


def main():
    """
    This function concatenates confirmation.csv and validation.csv provided
    from Trivago to form a ground truth file
    """
    
    df_conf = pd.read_csv(constants.INPUT_DIR / 'confirmation.csv')
    df_val = pd.read_csv(constants.INPUT_DIR / 'validation.csv')
    df_gt = pd.concat([df_conf, df_val], axis=0)
    df_gt.to_csv(constants.GROUND_TRUTH, index=False)
    print(f"Output saved to {constants.GROUND_TRUTH}.")


if __name__ == "__main__":
    main()
