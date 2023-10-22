import pandas as pd
import numpy as np
from scipy.stats import norm


def get_data(course_name, df):
    course_df = df[df[course_name].notnull()]

    if course_df.shape[0] == 0:
        return None, None, None, None, None, None

    course_mean = course_df[course_name].mean()
    course_stdev = course_df[course_name].std()

    mean_bounds = norm.interval(0.9, loc=course_mean, scale=(course_stdev/np.sqrt(course_df.shape[0])))

    course_df2 = course_df[course_df["Do you think the pace of teaching is too fast"
                                     " in any of the following modules?"].notnull()]

    too_fast_count = course_df2[
              course_df2["Do you think the pace of teaching is too fast in any of the following modules?"].apply(
                lambda row: course_name in row.split(";"))
                  ].shape[0]

    course_too_fast = too_fast_count / course_df.shape[0]

    too_fast_stdev = np.sqrt((too_fast_count * np.square(1 - course_too_fast)
                             + (course_df.shape[0] - too_fast_count) * np.square(course_too_fast))
                             / (course_df.shape[0] - 1))

    if too_fast_stdev == 0:
        too_fast_bounds = (np.nan, np.nan)
    else:
        too_fast_bounds = norm.interval(0.9, loc=course_too_fast, scale=(too_fast_stdev/np.sqrt(course_df.shape[0])))

    course_df3 = course_df[course_df[("Do you think the pace of teaching is too slow"
                                      " in any of the following modules?")].notnull()]

    too_slow_count = course_df3[
            course_df3["Do you think the pace of teaching is too slow in any of the following modules?"].apply(
                lambda row: course_name in row.split(";"))
                  ].shape[0]

    course_too_slow = too_slow_count / course_df.shape[0]

    too_slow_stdev = np.sqrt((too_slow_count * np.square(1 - course_too_slow)
                              + (course_df.shape[0] - too_slow_count) * np.square(course_too_slow))
                             / (course_df.shape[0] - 1))

    if too_slow_stdev == 0:
        too_slow_bounds = (np.nan, np.nan)
    else:
        too_slow_bounds = norm.interval(0.9, loc=course_too_slow, scale=(too_slow_stdev/np.sqrt(course_df.shape[0])))

    return course_mean, mean_bounds, course_too_fast, too_fast_bounds, course_too_slow, too_slow_bounds


df = pd.read_csv("Academic Feedback.csv")
df.drop("Timestamp", axis=1, inplace=True)
df.drop("What specifically have you enjoyed the most?", axis=1, inplace=True)
df.drop("Do you have any issues or concerns that you would like to address?", axis=1, inplace=True)

for course in ["CM12001 (Artificial Intelligence 1)", "CM12002 (Computer Systems Architectures)",
               "CM12003 (Programming 1)", "CM12004 (Discrete Mathematics and Databases)", "MA12012 (Algebra)",
               "MA12012 (Probability/Statistics)", "MA12012 (Sequences and Functions)"]:
    print(course)
    mean, mean_bounds, too_fast, too_fast_bounds, too_slow, too_slow_bounds = get_data(course, df)
    if mean is None:
        print("Mean score: no data")
    elif np.isnan(mean_bounds[0]):
        print(f"Too fast: {100 * mean:.1f}% (Can't calculate confidence interval)")
    else:
        print(f"Mean score: {mean:.1f} ({mean_bounds[0]:.1f}-{mean_bounds[1]:.1f})")
    if too_fast is None:
        print("Too fast: no data")
    elif np.isnan(too_fast_bounds[0]):
        print(f"Too fast: {100 * too_fast:.1f}% (Can't calculate confidence interval)")
    else:
        print(f"Too fast: {100 * too_fast:.1f}% "
              f"({max(0, 100 * too_fast_bounds[0]):.1f}%-{min(100, 100 * too_fast_bounds[1]):.1f}%)")
    if too_slow is None:
        print("Too slow: no data")
    elif np.isnan(too_slow_bounds[0]):
        print(f"Too fast: {100 * too_slow:.1f}% (Can't calculate confidence interval)")
    else:
        print(f"Too slow: {100 * too_slow:.1f}% "
              f"({max(0.0, 100 * too_slow_bounds[0]):.1f}%-{min(100.0, 100 * too_slow_bounds[1]):.1f}%)")

    print()
