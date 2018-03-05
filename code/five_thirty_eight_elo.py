# credit where credit is due
# https://www.kaggle.com/lpkirwin/fivethirtyeight-elo-ratings

import logging
import time

import numpy as np
import pandas as pd

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

default_head = 10
K = 21.0  # was 20.0
HOME_ADVANTAGE = 100.0


def elo_pred(arg_elo1, arg_elo2):
    result = (1.0 / (10.0 ** (-(arg_elo1 - arg_elo2) / 400.0) + 1.0))
    return result


def expected_margin(arg_elo_diff):
    result = (7.5 + 0.006 * arg_elo_diff)
    return result


def elo_update(aeg_w_elo, arg_l_elo, arg_margin, arg_k):
    local_pred = elo_pred(aeg_w_elo, arg_l_elo)
    multiplier = ((arg_margin + 3.0) ** 0.8) / expected_margin(aeg_w_elo - arg_l_elo)
    result = (local_pred, arg_k * multiplier * (1 - local_pred))
    return result


def final_elo_per_season(arg_df, arg_team_id):
    d = arg_df.copy()
    d = d.loc[(d.WTeamID == arg_team_id) | (d.LTeamID == arg_team_id), :]
    d.sort_values(['Season', 'DayNum'], inplace=True)
    d.drop_duplicates(['Season'], keep='last', inplace=True)
    w_mask = d.WTeamID == arg_team_id
    l_mask = d.LTeamID == arg_team_id
    d['season_elo'] = None
    d.loc[w_mask, 'season_elo'] = d.loc[w_mask, 'w_elo']
    d.loc[l_mask, 'season_elo'] = d.loc[l_mask, 'l_elo']
    result = pd.DataFrame({'team_id': arg_team_id, 'season': d.Season, 'season_elo': d.season_elo})
    return result


# todo add check for intput folder
input_file = '../input/RegularSeasonCompactResults.csv'
logger.debug('loading data from %s' % input_file)
rs = pd.read_csv(input_file)
logger.debug(rs.head(default_head))

team_ids = set(rs.WTeamID).union(set(rs.LTeamID))
logger.debug('we have %d team IDs' % len(team_ids))

# This dictionary will be used as a lookup for current
# scores while the algorithm is iterating through each game
elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))
# New columns to help us iteratively update elos
rs['margin'] = rs.WScore - rs.LScore
rs['w_elo'] = None
rs['l_elo'] = None

# I'm going to iterate over the games dataframe using
# index numbers, so want to check that nothing is out
# of order before I do that.
assert np.all(rs.index.values == np.array(range(rs.shape[0]))), "Index is out of order."

predictions = []

# Loop over all rows of the games dataframe
size = rs.shape[0]
increment = size / 1502
for i in range(size):

    # Get key data from current row
    w = rs.at[i, 'WTeamID']
    lx = rs.at[i, 'LTeamID']
    margin = rs.at[i, 'margin']
    wloc = rs.at[i, 'WLoc']

    # Does either team get a home-court advantage?
    w_ad, l_ad, = 0.0, 0.0
    if wloc == "H":
        w_ad += HOME_ADVANTAGE
    elif wloc == "A":
        l_ad += HOME_ADVANTAGE

    # Get elo updates as a result of the game
    prediction, update = elo_update(elo_dict[w] + w_ad, elo_dict[lx] + l_ad, margin, K)
    elo_dict[w] += update
    elo_dict[lx] -= update
    predictions.append(prediction)

    # Stores new elos in the games dataframe
    rs.loc[i, 'w_elo'] = elo_dict[w]
    rs.loc[i, 'l_elo'] = elo_dict[lx]
    if i % increment == 0:
        logger.debug('we have finished prediction %d of %d: (%.1f%%)' % (i, size, 100.0 * float(i) / float(size)))
logger.debug('done populating predictions.')

logger.debug(rs.tail(default_head))

logger.debug('fit: %.4f' % np.mean(-np.log(predictions)))

df_list = [final_elo_per_season(rs, i) for i in team_ids]
season_elos = pd.concat(df_list)

logger.debug(season_elos.sample(default_head))

# todo add check for output folder early
output_file = '../output/season_elos.csv'

season_elos.to_csv(output_file, index=None)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
