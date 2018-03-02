# credit where credit is due
# https://www.kaggle.com/virtonos/advanced-basketball-analytics

import logging
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

wtmp = df_season.apply(lambda row: row.WScore + row.WFGM + row.WFTM - row.WFGA - row.WFTA + row.WDR +
                                   0.5 * row.WOR + row.WAst + row.WStl + 0.5 * row.WBlk - row.WPF - row.WTO, axis=1)
ltmp = df_season.apply(lambda row: row.LScore + row.LFGM + row.LFTM - row.LFGA - row.LFTA + row.LDR +
                                   0.5 * row.LOR + row.LAst + row.LStl + 0.5 * row.LBlk - row.LPF - row.LTO, axis=1)

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

df_season['W4Factor'] = df_season.apply(lambda row: 0.40 * row.WeFGP + 0.25 * row.WTOR + 0.20 * row.WORP +
                                                    0.15 * row.WFTAR, axis=1)
df_season['L4Factor'] = df_season.apply(lambda row: 0.40 * row.LeFGP + 0.25 * row.LTOR + 0.20 * row.LORP +
                                                    0.15 * row.LFTAR, axis=1)

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

f, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(corrmatrix, vmax=.8, cbar=True, annot=True, square=True)
heatmap_file = '../output/correlation_heatmap.png'
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.savefig(heatmap_file)

df_rankings = pd.read_csv('../input/MasseyOrdinals.csv')
df_RPI = df_rankings[df_rankings['SystemName'] == 'RPI']
df_RPI_final = df_RPI[df_RPI['RankingDayNum'] == 133]
df_RPI_final.drop(labels=['RankingDayNum', 'SystemName'], inplace=True, axis=1)
logger.debug('RPI head: %s ' % df_RPI_final.head(default_head))

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))

# df_tourney = pd.read_csv('../input/NCAATourneyCompactResults.csv')
# df_teams = pd.read_csv('../input/Teams.csv')
# df_conferences = pd.read_csv('../input/Conferences.csv')
# df_sample_sub = pd.read_csv('../input/SampleSubmissionStage1.csv')
# df_seeds = pd.read_csv('../input/NCAATourneySeeds.csv')
