import pandas as pd

def __update_user_session_pairs(user_session_pairs, pairs_to_merge):
    """
    Return new user-sesion pair instance with merged pairs df
    """
    return user_session_pairs.merge(
        pairs_to_merge[['user_id', 'session_id']],
        left_on=['user_id', 'session_id'],
        right_on=['user_id', 'session_id'],
        how='outer'
    )


def __exclude_user_session_pairs(df, subset):
    """
    Return a new df window, which excludes the overlap of subset user-session pairs
    """
    mask = ~df.set_index(['user_id', 'session_id']).index.isin(subset.set_index(['user_id', 'session_id']).index)
    return df[mask]


def drop(df):
    # Create empty DF for storing user-session pairs to be deleted
    deleted_user_session_pairs = pd.DataFrame(columns=['user_id', 'session_id'])
    
    """
    Remove those users-sessions pairs where user has only one row (therefore single session and single action)
    """
    
    # Count number of actions (rows in df) of specific user
    user_id_count = df.groupby('user_id') \
        .agg({'session_id': 'first', 'user_id': 'count'}) \
        .rename(columns={'user_id': 'count'}) \
        .reset_index()[['user_id', 'session_id', 'count']]
    
    users_with_one_action = user_id_count[(user_id_count['count'] == 1)]
    
    # Drop
    deleted_user_session_pairs = __update_user_session_pairs(deleted_user_session_pairs, users_with_one_action)
    df = __exclude_user_session_pairs(df, deleted_user_session_pairs)
    
    """
    Remove those user-session pairs where session has only one row (therefore single action)
    """
    # Count number of actions (rows in df) of specific session
    session_action_count = df.groupby('session_id') \
        .agg({'user_id': 'first', 'session_id': 'count'}) \
        .rename(columns={'session_id': 'count'}) \
        .reset_index()[['user_id', 'session_id', 'count']]
    
    sessions_with_one_action = session_action_count[(session_action_count['count'] == 1)]
    
    # Drop
    deleted_user_session_pairs = __update_user_session_pairs(deleted_user_session_pairs, sessions_with_one_action)
    df = __exclude_user_session_pairs(df, deleted_user_session_pairs)
    
    """
    Remove those user-session pairs where session has more than TRESHOLD rows
    """
    excessive_threshold = 500
    sessions_with_excessive_actions = session_action_count[(session_action_count['count'] > excessive_threshold)]
    
    # Drop
    deleted_user_session_pairs = __update_user_session_pairs(deleted_user_session_pairs, sessions_with_excessive_actions)
    df = __exclude_user_session_pairs(df, deleted_user_session_pairs)

    """
    Remove those user-session pairs where session has duplicated steps )e.g. one session has multiple rows with same index)
    """
    duped_user_sessions = df[(df.duplicated(subset=['user_id', 'session_id', 'step'], keep=False))]\
        .groupby('session_id')\
        .agg({'user_id': 'first', 'session_id': 'first'})\
        .reset_index(drop=True)[['user_id', 'session_id']]
    
    # Drop
    deleted_user_session_pairs = __update_user_session_pairs(deleted_user_session_pairs, duped_user_sessions)
    df = __exclude_user_session_pairs(df, deleted_user_session_pairs)
    
    
    """
    Remove those user-session pairs where session has duplicated steps )e.g. one session has multiple rows with same index)
    """
    def min_max_diff(x):
        return x.max() - x.min()
    
    # Get duration of session
    u_actions_per_session = df.groupby(['user_id', 'session_id'], as_index=False) \
        .agg(
        steps=('step', 'count'),
        first_timestamp=('timestamp', 'first'),
        last_timestamp=('timestamp', 'last'),
        session_duration=('timestamp', min_max_diff),
    )
    
    def avg_step_duration(row):
        return row['session_duration'] / row['steps']

    def seconds_to_minutes(seconds):
        return round(seconds / 60, 2)

    # Transform session length into minutes
    u_actions_per_session['session_duration_minutes'] = u_actions_per_session['session_duration'].apply(seconds_to_minutes);

    # Calculate average duration of step
    u_actions_per_session['avg_step_duration_seconds'] = u_actions_per_session.apply(avg_step_duration, axis=1);

    step_duration_minimum = 1
    users_sessions_avg_step_under_threshold = u_actions_per_session[u_actions_per_session['avg_step_duration_seconds'] < step_duration_minimum]
    
    # Drop
    deleted_user_session_pairs = __update_user_session_pairs(deleted_user_session_pairs, users_sessions_avg_step_under_threshold)
    df = __exclude_user_session_pairs(df, deleted_user_session_pairs)
    
    return df


