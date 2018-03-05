# credit where credit is due
# https://www.kaggle.com/virtonos/advanced-basketball-analytics

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

# from xgboost.sklearn import XGBClassifier

pd.set_option('display.max_columns', 999)

start_time = time.time()


# Return a tuple with ints `year`, `team1` and `team2`.
def get_year_t1_t2(arg_id):
    result = (int(item) for item in arg_id.split('_'))
    return result


formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

default_head = 10

input_files = sorted([
    '../input/NCAATourneyCompactResults.csv',
    '../input/RegularSeasonDetailedResults.csv',
    '../input/Teams.csv',
    '../input/NCAATourneySeeds.csv',
    '../input/Conferences.csv',
    '../input/MasseyOrdinals.csv',
    '../input/SampleSubmissionStage1.csv'
])
for input_file in input_files:
    if os.path.isfile(input_file):
        logger.debug('input file %s exists' % input_file)
    else:
        logger.warning('input file %s is missing. Quitting.' % input_file)

df_season = pd.read_csv('../input/RegularSeasonDetailedResults.csv')

logger.debug('season data has shape %d x %d' % df_season.shape)
# Calculate Winning/losing Team Possession Feature
# https://www.nbastuffer.com/analytics101/possession/
wPos = df_season.apply(lambda row: 0.96 * (row.WFGA + row.WTO + 0.44 * row.WFTA - row.WOR), axis=1)
lPos = df_season.apply(lambda row: 0.96 * (row.LFGA + row.LTO + 0.44 * row.LFTA - row.LOR), axis=1)

# two teams use almost the same number of possessions in a game
# (plus/minus one or two - depending on how quarters end)
# so let's just take the average

df_season['Possessions'] = (wPos + lPos) / 2
logger.debug(df_season['Possessions'].head(default_head))

# Name Player Impact Estimate Definition PIE measures a player's overall statistical contribution
# against the total statistics in games they play in. PIE yields results which are
# comparable to other advanced statistics (e.g. PER) using a simple formula.
# Formula (PTS + FGM + FTM - FGA - FTA + DREB + (.5 * OREB) + AST + STL + (.5 * BLK) - PF - TO)
# / (GmPTS + GmFGM + GmFTM - GmFGA - GmFTA + GmDREB + (.5 * GmOREB) + GmAST + GmSTL + (.5 * GmBLK) - GmPF - GmTO)

wtmp = df_season.apply(lambda row:
                       row.WScore + row.WFGM + row.WFTM - row.WFGA - row.WFTA + row.WDR + 0.5 * row.WOR +
                       row.WAst + row.WStl + 0.5 * row.WBlk - row.WPF - row.WTO,
                       axis=1)
ltmp = df_season.apply(lambda
                           row:
                       row.LScore + row.LFGM + row.LFTM - row.LFGA - row.LFTA + row.LDR + 0.5 * row.LOR +
                       row.LAst + row.LStl + 0.5 * row.LBlk - row.LPF - row.LTO,
                       axis=1)

df_season['WPIE'] = wtmp / (wtmp + ltmp)
df_season['LPIE'] = ltmp / (wtmp + ltmp)

# Four factors statistic from the NBA

# https://www.nbastuffer.com/analytics101/four-factors/


# Effective Field Goal Percentage=(Field Goals Made) + 0.5*3P Field Goals Made))/(Field Goal Attempts)
# you have to put the ball in the bucket eventually

df_season['WeFGP'] = df_season.apply(lambda row: (row.WFGM + 0.5 * row.WFGM3) / row.WFGA, axis=1)
df_season['LeFGP'] = df_season.apply(lambda row: (row.LFGM + 0.5 * row.LFGM3) / row.LFGA, axis=1)

# Turnover Rate= Turnovers/(Field Goal Attempts + 0.44*Free Throw Attempts + Turnovers)
# he who doesnt turn the ball over wins games

df_season['WTOR'] = df_season.apply(lambda row: row.WTO / (row.WFGA + 0.44 * row.WFTA + row.WTO), axis=1)
df_season['LTOR'] = df_season.apply(lambda row: row.LTO / (row.LFGA + 0.44 * row.LFTA + row.LTO), axis=1)

# Offensive Rebounding Percentage = (Offensive Rebounds)/[(Offensive Rebounds)+(Opponent’s Defensive Rebounds)]
# You can win games controlling the offensive glass

df_season['WORP'] = df_season.apply(lambda row: row.WOR / (row.WOR + row.LDR), axis=1)
df_season['LORP'] = df_season.apply(lambda row: row.LOR / (row.LOR + row.WDR), axis=1)

# Free Throw Rate=(Free Throws Made)/(Field Goals Attempted) or Free Throws Attempted/Field Goals Attempted
# You got to get to the line to win close games

df_season['WFTAR'] = df_season.apply(lambda row: row.WFTA / row.WFGA, axis=1)
df_season['LFTAR'] = df_season.apply(lambda row: row.LFTA / row.LFGA, axis=1)

# 4 Factors is weighted as follows
# 1. Shooting (40%)
# 2. Turnovers (25%)
# 3. Rebounding (20%)
# 4. Free Throws (15%)

df_season['W4Factor'] = df_season.apply(
    lambda row: 0.40 * row.WeFGP + 0.25 * row.WTOR + 0.20 * row.WORP + 0.15 * row.WFTAR, axis=1)
df_season['L4Factor'] = df_season.apply(
    lambda row: 0.40 * row.LeFGP + 0.25 * row.LTOR + 0.20 * row.LORP + 0.15 * row.LFTAR, axis=1)

# Offensive efficiency (OffRtg) =  (Points / Possessions)
# Every possession counts

df_season['WOffRtg'] = df_season.apply(lambda row: (row.WScore / row.Possessions), axis=1)
df_season['LOffRtg'] = df_season.apply(lambda row: (row.LScore / row.Possessions), axis=1)

# Defensive efficiency (DefRtg) = (Opponent points / Opponent possessions)
# defense wins championships

df_season['WDefRtg'] = df_season.LOffRtg
df_season['LDefRtg'] = df_season.WOffRtg

# Assist Ratio : Percentage of team possessions that end in assists
# distribute the rock - don't go isolation all the time

df_season['WAstR'] = df_season.apply(lambda row: row.WAst / (row.WFGA + 0.44 * row.WFTA + row.WAst + row.WTO), axis=1)
df_season['LAstR'] = df_season.apply(lambda row: row.LAst / (row.LFGA + 0.44 * row.LFTA + row.LAst + row.LTO), axis=1)

# DREB% : Percentage of team defensive rebounds
# control your own glass

df_season['WDRP'] = df_season.apply(lambda row: row.WDR / (row.WDR + row.LOR), axis=1)
df_season['LDRP'] = df_season.apply(lambda row: row.LDR / (row.LDR + row.WOR), axis=1)

# Free Throw Percentage
# Make your free throws

df_season['WFTPCT'] = df_season.apply(lambda row: 0 if row.WFTA < 1 else row.WFTM / row.WFTA, axis=1)
df_season['LFTPCT'] = df_season.apply(lambda row: 0 if row.LFTA < 1 else row.LFTM / row.LFTA, axis=1)

season_columns_to_drop = sorted(
    ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk',
     'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
     'LBlk', 'LPF'])
logger.debug('dropping season columns %s' % season_columns_to_drop)

df_season.drop(season_columns_to_drop, axis=1, inplace=True)

df_season_composite = pd.DataFrame()

# This will aggregate individual games into season totals for a team

# calculates wins and losses to get winning percentage

df_season_composite['WINS'] = df_season['WTeamID'].groupby([df_season['Season'], df_season['WTeamID']]).count()
df_season_composite['LOSSES'] = df_season['LTeamID'].groupby([df_season['Season'], df_season['LTeamID']]).count()
df_season_composite['WINPCT'] = df_season_composite['WINS'] / (df_season_composite['WINS'] +
                                                               df_season_composite['LOSSES'])

# calculates averages for games team won

df_season_composite['WPIE'] = df_season['WPIE'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
df_season_composite['WeFGP'] = df_season['WeFGP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
df_season_composite['WTOR'] = df_season['WTOR'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
df_season_composite['WORP'] = df_season['WORP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
df_season_composite['WFTAR'] = df_season['WFTAR'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
df_season_composite['W4Factor'] = df_season['W4Factor'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
df_season_composite['WOffRtg'] = df_season['WOffRtg'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
df_season_composite['WDefRtg'] = df_season['WDefRtg'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
df_season_composite['WAstR'] = df_season['WAstR'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
df_season_composite['WDRP'] = df_season['WDRP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
df_season_composite['WFTPCT'] = df_season['WFTPCT'].groupby([df_season['Season'], df_season['WTeamID']]).mean()

# calculates averages for games team lost

df_season_composite['LPIE'] = df_season['LPIE'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
df_season_composite['LeFGP'] = df_season['LeFGP'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
df_season_composite['LTOR'] = df_season['LTOR'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
df_season_composite['LORP'] = df_season['LORP'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
df_season_composite['LFTAR'] = df_season['LFTAR'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
df_season_composite['L4Factor'] = df_season['L4Factor'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
df_season_composite['LOffRtg'] = df_season['LOffRtg'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
df_season_composite['LDefRtg'] = df_season['LDefRtg'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
df_season_composite['LAstR'] = df_season['LAstR'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
df_season_composite['LDRP'] = df_season['LDRP'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
df_season_composite['LFTPCT'] = df_season['LFTPCT'].groupby([df_season['Season'], df_season['LTeamID']]).mean()

# calculates weighted average using winning percent to weight the statistic


df_season_composite['PIE'] = df_season_composite['WPIE'] * df_season_composite['WINPCT'] + \
                             df_season_composite['LPIE'] * (1 - df_season_composite['WINPCT'])
df_season_composite['FG_PCT'] = df_season_composite['WeFGP'] * df_season_composite['WINPCT'] + \
                                df_season_composite['LeFGP'] * (1 - df_season_composite['WINPCT'])
df_season_composite['TURNOVER_RATE'] = df_season_composite['WTOR'] * df_season_composite['WINPCT'] + \
                                       df_season_composite['LTOR'] * (1 - df_season_composite['WINPCT'])
df_season_composite['OFF_REB_PCT'] = df_season_composite['WORP'] * df_season_composite['WINPCT'] + \
                                     df_season_composite['LORP'] * (1 - df_season_composite['WINPCT'])
df_season_composite['FT_RATE'] = df_season_composite['WFTAR'] * df_season_composite['WINPCT'] + \
                                 df_season_composite['LFTAR'] * (1 - df_season_composite['WINPCT'])
df_season_composite['4FACTOR'] = df_season_composite['W4Factor'] * df_season_composite['WINPCT'] + \
                                 df_season_composite['L4Factor'] * (1 - df_season_composite['WINPCT'])
df_season_composite['OFF_EFF'] = df_season_composite['WOffRtg'] * df_season_composite['WINPCT'] + \
                                 df_season_composite['LOffRtg'] * (1 - df_season_composite['WINPCT'])
df_season_composite['DEF_EFF'] = df_season_composite['WDefRtg'] * df_season_composite['WINPCT'] + \
                                 df_season_composite['LDefRtg'] * (1 - df_season_composite['WINPCT'])
df_season_composite['ASSIST_RATIO'] = df_season_composite['WAstR'] * df_season_composite['WINPCT'] + \
                                      df_season_composite['LAstR'] * (1 - df_season_composite['WINPCT'])
df_season_composite['DEF_REB_PCT'] = df_season_composite['WDRP'] * df_season_composite['WINPCT'] + \
                                     df_season_composite['LDRP'] * (1 - df_season_composite['WINPCT'])
df_season_composite['FT_PCT'] = df_season_composite['WFTPCT'] * df_season_composite['WINPCT'] + \
                                df_season_composite['LFTPCT'] * (1 - df_season_composite['WINPCT'])

df_season_composite.reset_index(inplace=True)

df_season_composite.loc[4064, 'WINPCT'] = 1
df_season_composite.loc[4064, 'LOSSES'] = 0
df_season_composite.loc[4064, 'PIE'] = df_season_composite.loc[4064, 'WPIE']
df_season_composite.loc[4064, 'FG_PCT'] = df_season_composite.loc[4064, 'WeFGP']
df_season_composite.loc[4064, 'TURNOVER_RATE'] = df_season_composite.loc[4064, 'WTOR']
df_season_composite.loc[4064, 'OFF_REB_PCT'] = df_season_composite.loc[4064, 'WORP']
df_season_composite.loc[4064, 'FT_RATE'] = df_season_composite.loc[4064, 'WFTAR']
df_season_composite.loc[4064, '4FACTOR'] = df_season_composite.loc[4064, 'W4Factor']
df_season_composite.loc[4064, 'OFF_EFF'] = df_season_composite.loc[4064, 'WOffRtg']
df_season_composite.loc[4064, 'DEF_EFF'] = df_season_composite.loc[4064, 'WDefRtg']
df_season_composite.loc[4064, 'ASSIST_RATIO'] = df_season_composite.loc[4064, 'WAstR']
df_season_composite.loc[4064, 'DEF_REB_PCT'] = df_season_composite.loc[4064, 'WDRP']
df_season_composite.loc[4064, 'FT_PCT'] = df_season_composite.loc[4064, 'WFTPCT']

df_season_composite.loc[4211, 'WINPCT'] = 1
df_season_composite.loc[4211, 'LOSSES'] = 0
df_season_composite.loc[4211, 'PIE'] = df_season_composite.loc[4211, 'WPIE']
df_season_composite.loc[4211, 'FG_PCT'] = df_season_composite.loc[4211, 'WeFGP']
df_season_composite.loc[4211, 'TURNOVER_RATE'] = df_season_composite.loc[4211, 'WTOR']
df_season_composite.loc[4211, 'OFF_REB_PCT'] = df_season_composite.loc[4211, 'WORP']
df_season_composite.loc[4211, 'FT_RATE'] = df_season_composite.loc[4211, 'WFTAR']
df_season_composite.loc[4211, '4FACTOR'] = df_season_composite.loc[4211, 'W4Factor']
df_season_composite.loc[4211, 'OFF_EFF'] = df_season_composite.loc[4211, 'WOffRtg']
df_season_composite.loc[4211, 'DEF_EFF'] = df_season_composite.loc[4211, 'WDefRtg']
df_season_composite.loc[4211, 'ASSIST_RATIO'] = df_season_composite.loc[4211, 'WAstR']
df_season_composite.loc[4211, 'DEF_REB_PCT'] = df_season_composite.loc[4211, 'WDRP']
df_season_composite.loc[4211, 'FT_PCT'] = df_season_composite.loc[4211, 'WFTPCT']

# we only need the final summary stats
season_composite_columns_to_drop = sorted(
    ['WINS', 'WPIE', 'WeFGP', 'WTOR', 'WORP', 'WFTAR', 'W4Factor', 'WOffRtg', 'WDefRtg', 'WAstR', 'WDRP', 'WFTPCT',
     'LOSSES', 'LPIE', 'LeFGP', 'LTOR', 'LORP', 'LFTAR', 'L4Factor', 'LOffRtg', 'LDefRtg', 'LAstR', 'LDRP', 'LFTPCT'])
logger.debug('dropping season composite columns %s' % season_composite_columns_to_drop)
df_season_composite.drop(season_composite_columns_to_drop, axis=1, inplace=True)

columns = list(df_season_composite.columns.values)
columns.pop(columns.index('WINPCT'))
columns.append('WINPCT')
df_season_composite = df_season_composite[columns]
df_season_composite.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
logger.debug('season composite head: %s' % df_season_composite.head(default_head))

logger.debug('done building season composite data frame')

corrmatrix = df_season_composite.iloc[:, 2:].corr()

figure, axes = plt.subplots(figsize=(11, 7))
sns.heatmap(corrmatrix, vmax=.8, cbar=True, annot=True, square=True)
heatmap_file = '../output/correlation_heatmap.png'
logger.debug('writing correlation matrix to %s' % heatmap_file)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.savefig(heatmap_file)
del figure
del axes

df_rankings = pd.read_csv('../input/MasseyOrdinals.csv')
logger.debug('rankings data frame size: %d x %d' % df_rankings.shape)

df_RPI = df_rankings[df_rankings['SystemName'] == 'RPI']
df_RPI_final = df_RPI[df_RPI['RankingDayNum'] == 133]
df_RPI_final.drop(labels=['RankingDayNum', 'SystemName'], inplace=True, axis=1)
logger.debug('RPI head: %s ' % df_RPI_final.head(default_head))

df_seeds = pd.read_csv('../input/NCAATourneySeeds.csv')
logger.debug('seeds data frame size: %d x %d' % df_seeds.shape)

logger.debug(df_seeds.head(default_head))
# Convert string to an integer

df_seeds['seed_int'] = df_seeds['Seed'].apply(lambda x: int(x[1:3]))
df_seeds.drop(labels=['Seed'], inplace=True, axis=1)
df_seeds.rename(columns={'seed_int': 'Seed'}, inplace=True)
df_seeds.head(default_head)

# Create dataframe of team features for all seasons

# ranks only start since 2003

df_seeds_final = df_seeds[df_seeds['Season'] > 2002]

# 2 step merge

tourney_merge_columns = ['Season', 'TeamID']
df_tourney_stage = pd.merge(left=df_seeds_final, right=df_RPI_final, how='left', on=tourney_merge_columns)
df_tourney_final = pd.merge(left=df_tourney_stage, right=df_season_composite, how='left', on=tourney_merge_columns)
logger.debug(df_tourney_final.head(default_head))

df_tourney = pd.read_csv('../input/NCAATourneyCompactResults.csv')
logger.debug('tournament data frame shape: %d x %d' % df_tourney.shape)

df_tourney.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_tourney = pd.merge(left=df_tourney, right=df_seeds, how='left', left_on=['Season', 'WTeamID'],
                      right_on=tourney_merge_columns)
df_tourney = pd.merge(left=df_tourney, right=df_seeds, how='left', left_on=['Season', 'LTeamID'],
                      right_on=tourney_merge_columns)
df_tourney.drop(labels=['TeamID_x', 'TeamID_y'], inplace=True, axis=1)
df_tourney.rename(columns={'Seed_x': 'WSeed', 'Seed_y': 'LSeed'}, inplace=True)
logger.debug(df_tourney.head(default_head))

df_tourney['SeedDiff'] = df_tourney['WSeed'] - df_tourney['LSeed']
figure, axes = plt.subplots(figsize=(11, 7))
sns.countplot(df_tourney['SeedDiff'])
countplot_file = '../output/countplot.png'
logger.debug('writing seed-difference count plot to %s' % countplot_file)
plt.savefig(countplot_file)
del figure
del axes

df_wins = pd.DataFrame()
df_wins['SeedDiff'] = df_tourney['SeedDiff']
df_wins['Result'] = 1

df_losses = pd.DataFrame()
df_losses['SeedDiff'] = -df_tourney['SeedDiff']
df_losses['Result'] = 0

df_predictions = pd.concat([df_wins, df_losses])
logger.debug('predictions shape is %d x %d' % df_predictions.shape)
logger.debug('predictions: %s' % df_predictions.head(default_head))

# setup the data

X_train = df_predictions.SeedDiff.values.reshape(-1, 1)
y_train = df_predictions.Result.values
X_train, y_train = shuffle(X_train, y_train)

# use Logistic regression with Gridsearch for parameter tuning

logreg = LogisticRegression(random_state=0)
params = {'C': np.logspace(start=-5, stop=3, num=9)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True, cv=10)
clf.fit(X_train, y_train)
logger.debug('Best log_loss: %.4f, with best C: %.3f' % (clf.best_score_, clf.best_params_['C']))

X = np.arange(-15, 15).reshape(-1, 1)  # this creates the range of seed differentials
logreg_preds = clf.predict_proba(X)[:, 1]  # the 1 signifies winning

figure, axes = plt.subplots(figsize=(11, 7))
plt.plot(X, logreg_preds)
plt.xlabel('Team1 seed - Team2 seed')
plt.ylabel('P(Team1 will win)')
seed_seed_graph_file = '../output/seed_seed_graph.png'
plt.savefig(seed_seed_graph_file)
del figure
del axes

train_acc = accuracy_score(y_true=y_train, y_pred=clf.predict(X_train))
logger.debug('Training Accuracy: %.2f%%' % (100 * train_acc))

df_sample_sub = pd.read_csv('../input/SampleSubmissionStage1.csv')
logger.debug('sample sub data frame is %d x %d' % df_sample_sub.shape)
logger.debug('sample sub head : %s' % df_sample_sub.head(default_head))

# This generates a submission file for 2014-2017 using the simple Seeds model
n_test_games = len(df_sample_sub)
logger.debug('n test games : %d' % n_test_games)
X_test = np.zeros(shape=(n_test_games, 1))
logger.debug('X test shape is %d x %d' % X_test.shape)

for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    team1 = df_tourney_final[(df_tourney_final.TeamID == t1) & (df_tourney_final.Season == year)].Seed.values[0]
    team2 = df_tourney_final[(df_tourney_final.TeamID == t2) & (df_tourney_final.Season == year)].Seed.values[0]
    diff_seed = team1 - team2
    X_test[ii, 0] = diff_seed

seeds_preds = clf.predict_proba(X_test)[:, 1]

logger.debug('df sample sub shape: %d x %d' % df_sample_sub.shape)
logger.debug('X_test shape: %d x %d' % X_test.shape)
logger.debug('preds length: %d' % len(seeds_preds))
df_sample_sub['Pred'] = seeds_preds
logger.debug('sample sub shape: %d x %d' % df_sample_sub.shape)
logreg_file = '../output/logreg_seed_starter.csv'
logger.debug('writing sample sub to %s' % logreg_file)
df_sample_sub.to_csv(logreg_file, index=False)
logger.debug('sample sub head: %s' % df_sample_sub.head(default_head))

logger.debug('tourney final head: %s' % df_tourney_final.head(default_head))

df_tourney_list = pd.read_csv('../input/NCAATourneyCompactResults.csv')
df_tourney_list.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_tourney_list = df_tourney_list[df_tourney_list['Season'] > 2002]
df_tourney_list.reset_index(inplace=True, drop=True)
logger.debug('tourney list (head): %s' % df_tourney_list.head(default_head))

# gets the features for the winning team

df_model_winners = pd.merge(left=df_tourney_list, right=df_tourney_final, how='left', left_on=['Season', 'WTeamID'],
                            right_on=['Season', 'TeamID'])
df_model_winners.drop(labels=['TeamID'], inplace=True, axis=1)
logger.debug('model winners (head): %s' % df_model_winners.head(default_head))

# gets the features for the losing team

df_model_losers = pd.merge(left=df_tourney_list, right=df_tourney_final, how='left', left_on=['Season', 'LTeamID'],
                           right_on=['Season', 'TeamID'])
df_model_losers.drop(labels=['TeamID'], inplace=True, axis=1)
logger.debug('model losers (head): %s' % df_model_losers.head(default_head))

# This generates the differences between the features between winning and losing team and assigns 1 as the classifier for winning

df_model_winner_diff = (df_model_winners.iloc[:, 3:] - df_model_losers.iloc[:, 3:])
df_model_winner_diff['result'] = 1
df_model_winner_diff = pd.merge(left=df_model_winner_diff, right=df_tourney_list, left_index=True, right_index=True,
                                how='inner')

logger.debug('model winners (head): %s' % df_model_winner_diff.head(default_head))
# This generates the  between the features between losing and winning team and assigns 0 as the classifier for losing

df_model_loser_diff = (df_model_losers.iloc[:, 3:] - df_model_winners.iloc[:, 3:])
df_model_loser_diff['result'] = 0
df_model_loser_diff = pd.merge(left=df_model_loser_diff, right=df_tourney_list, left_index=True, right_index=True,
                               how='inner')
logger.debug('model losers (head): %s' % df_model_loser_diff.head(default_head))

df_predictions_tourney = pd.concat((df_model_winner_diff, df_model_loser_diff), axis=0)
df_predictions_tourney.sort_values('Season', inplace=True)
df_predictions_tourney.reset_index(inplace=True, drop=True)
logger.debug('tourney predictions (head): %s' % df_predictions_tourney.head(default_head))

# Time to split the dataframe into its various components for ML
labels = df_predictions_tourney['result']
IDs = df_predictions_tourney.iloc[:, 15:]
features = df_predictions_tourney.iloc[:, 0:14]
logger.debug('features (head): %s' % features.head(default_head))

# This will be the true test set for submission for phase one (4 years 2014-2017)

labels_submission = df_predictions_tourney['result'][df_predictions_tourney['Season'] > 2013]
IDs_submission = df_predictions_tourney.iloc[1426:, 15:]
features_submission = df_predictions_tourney.iloc[1426:, 0:14]

# This will be the training set

y = df_predictions_tourney['result'][df_predictions_tourney['Season'] < 2014]
IDs_training = df_predictions_tourney.iloc[:1426, 15:]
X = df_predictions_tourney.iloc[:1426, 0:14]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1, stratify=y)

clf1 = LogisticRegression()

clf2 = KNeighborsClassifier()

# clf3 = XGBClassifier()

clf4 = DecisionTreeClassifier()

clf5 = RandomForestClassifier()

clf6 = GradientBoostingClassifier()

# Setting up the parameter grids

param_grid1 = [{'clf1__C': list(np.logspace(start=-5, stop=3, num=9))}]

param_grid2 = [
    {'clf2__n_neighbors': list(range(1, 20)), 'clf2__p': [1, 2], 'clf2__algorithm': ['ball_tree', 'kd_tree']}]

# param_grid3 = [{'learning_rate' : [0.1, 0.3],
#                 'max_depth': [3, 6],
#                 'min_child_weight': list(range(1, 3))}]

param_grid4 = [{'max_depth': list(range(3, 6)), 'criterion': ['gini', 'entropy'], 'min_samples_leaf': [20, 50]}]

param_grid5 = [{'max_depth': list(range(1, 5)), 'criterion': ['gini', 'entropy'], 'min_samples_split': [2, 3]}]

param_grid6 = [{'learning_rate': [0.01, 0.1], 'loss': ['deviance', 'exponential'], 'max_depth': list(range(3, 4))}]

# Building the pipelines
pipe1 = Pipeline([('std', StandardScaler()), ('clf1', clf1)])
pipe2 = Pipeline([('std', StandardScaler()), ('clf2', clf2)])
# pipe3 = Pipeline([('std', StandardScaler()),('clf3', clf3)])
# pipe4 = Pipeline([('std', StandardScaler()), ('clf4', clf4)])
# pipe5 = Pipeline([('std', StandardScaler()), ('clf5', clf5)])
# pipe6 = Pipeline([('std', StandardScaler()), ('clf6', clf6)])

# Setting up multiple GridSearchCV objects, 1 for each algorithm

gridcvs = {}

inner_cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=2)
outer_cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=2)

for pgrid, est, name in zip((param_grid1, param_grid2,
                             # param_grid3,
                             param_grid4,
                             param_grid5, param_grid6),
                            (pipe1, pipe2,
                             # clf3,
                             clf4, clf5, clf6),
                            ('Logistic', 'KNN',
                             # 'XGBoost',
                             'DTree', 'Random Forest', 'Gradient Boosting')):

    # (pipe1, pipe2, clf3, clf4, clf5, clf6),
    # First loop runs GridSearch and does Cross validation to find the best parameters

    gcv = GridSearchCV(estimator=est,
                       param_grid=pgrid,
                       scoring='neg_log_loss',
                       cv=outer_cv,
                       verbose=0,
                       refit=True,
                       return_train_score=False)

    gcv.fit(X_train, y_train)

    gridcvs[name] = gcv

    logger.debug('model %s has best estimator %s' % (name, gcv.best_estimator_))
    logger.debug('Best score on Grid Search Cross Validation is %.2f%%' % (gcv.best_score_))
    # print('Accuracy %.2f%% (average over CV test folds)' % (100 * best_algo.best_score_))
    results = pd.DataFrame(gcv.cv_results_)

    # Inner loop runs Cross Val Score on tuned parameter model to determine accuracy of fit

    # for name, gs_est in sorted(gridcvs.items()):

    nested_score = cross_val_score(gcv,
                                   X=X_train,
                                   y=y_train,
                                   cv=inner_cv,
                                   scoring='neg_log_loss')

    logger.debug('Name, Log Loss, Std Dev, based on Best Parameter Model using Cross Validation Scoring')
    logger.debug('%s | %.2f %.2f' % (name, nested_score.mean(), nested_score.std() * 100))

    # Generate predictions and probabilities
    best_algo = gcv
    best_algo.fit(X_train, y_train)

    train_acc = accuracy_score(y_true=y_train, y_pred=best_algo.predict(X_train))
    test_acc = accuracy_score(y_true=y_test, y_pred=best_algo.predict(X_test))

    logger.debug('Training Accuracy: %.2f%%' % (100 * train_acc))
    logger.debug('Test Accuracy: %.2f%%' % (100 * test_acc))
    # print('Slippage: %.2f%%' % ((100 * test_acc) - (100 * train_acc)))

    # prints classification report and confusion matrix

    if name != 'SVM':

        predictions = best_algo.predict(X_test)
        probability = best_algo.predict_proba(X_test)
        logger.debug(classification_report(y_test, predictions))
        logger.debug(confusion_matrix(y_test, predictions))

    else:
        logger.debug('for SVM we have no classification report or confusion matrix')

# retrain best classifier on the full scaled training data set now

clf = LogisticRegression(C=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf.fit(X_scaled, y)

logger.debug('X (head) : %s' % X.head(default_head))

# steps for grabbing teams info and creating the input for the model

n_test_games = len(df_sample_sub)
X_test = np.zeros(shape=(n_test_games, 1))
columns = df_tourney_final.columns.get_values()
# model = []
data = []

for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    team1 = df_tourney_final[(df_tourney_final.TeamID == t1) & (df_tourney_final.Season == year)].values
    team2 = df_tourney_final[(df_tourney_final.TeamID == t2) & (df_tourney_final.Season == year)].values
    # model =
    data.append(team1 - team2)

Predictions = pd.DataFrame(np.array(data).reshape(9112, 16), columns=(columns))
Predictions.drop(labels=['Season', 'TeamID'], inplace=True, axis=1)
logger.debug('predictions (head): %s', Predictions.head(default_head))

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))

# df_teams = pd.read_csv('../input/Teams.csv')
# df_conferences = pd.read_csv('../input/Conferences.csv')
