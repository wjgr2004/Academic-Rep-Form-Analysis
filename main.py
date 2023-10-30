import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as tck


plt.style.use("seaborn-v0_8-whitegrid")
plt.grid(linewidth=0.8, color="darkgrey")
plt.grid(linewidth=0.4, color="gainsboro", which="minor")


def get_data(course_name, df):
    course_df = df[df[course_name].notnull()]

    if course_df.shape[0] == 0:
        return None, None, None, None, None, None, None

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    ax.tick_params(which="major", length=5, width=1.3)

    ax.set_xticks(np.arange(1, 11))

    ax.set_title(f"{course_name} Ratings", loc="left")
    ax.set_xlabel("Rating")

    data = course_df[course_name].value_counts()

    ax.set(xlim=(0.5, 10.5), ylim=(0, data.max() * 1.1))

    ax.yaxis.set_major_locator(tck.MaxNLocator(integer=True))

    ax.bar(data.index, data)

    with open(f"Graphs/{course_name.replace('/', ',')}.png", "wb") as file:
        plt.savefig(file)

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

    return (course_mean, mean_bounds, course_too_fast, too_fast_bounds,
            course_too_slow, too_slow_bounds, course_df.shape[0])


df = pd.read_csv("Academic Feedback.csv")
df.drop("Timestamp", axis=1, inplace=True)
df.drop("What specifically have you enjoyed the most?", axis=1, inplace=True)
df.drop("Do you have any issues or concerns that you would like to address?", axis=1, inplace=True)

means = []
mean_bounds_list = []

too_slows = []
too_slow_bounds_list = []

too_fasts = []
too_fast_bounds_list = []

course_names = ["CM12001 (Artificial Intelligence 1)", "CM12002 (Computer Systems Architectures)",
               "CM12003 (Programming 1)", "CM12004 (Discrete Mathematics and Databases)", "MA12012 (Algebra)",
               "MA12012 (Probability/Statistics)", "MA12012 (Sequences and Functions)"]

short_course_names = ["AI", "Sys Arch", "Programming", "Discrete Maths", "Algebra", "Statistics", "Sequences"]

for course in course_names:
    print(course)
    mean, mean_bounds, too_fast, too_fast_bounds, too_slow, too_slow_bounds, responses = get_data(course, df)
    means.append(mean)
    mean_bounds_list.append(mean_bounds)
    too_slows.append(too_slow)
    too_slow_bounds_list.append(too_slow_bounds)
    too_fasts.append(too_fast)
    too_fast_bounds_list.append(too_fast_bounds)
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
    if responses is None:
        print("Responses: no data")
    else:
        print(f"Responses: {responses}")

    print()

means = np.array(means)
mean_bounds_list = np.array(mean_bounds_list).transpose()

too_slows = np.array(too_slows) * 100
too_slow_bounds_list = np.array(too_slow_bounds_list).transpose() * 100

too_fasts = np.array(too_fasts) * 100
too_fast_bounds_list = np.array(too_fast_bounds_list).transpose() * 100

fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

ax.tick_params(which="major", length=5, width=1.3)

ax.set_title(f"Course Ratings", loc="left")
ax.set_xlabel("Course")
ax.set_ylabel("Rating")

# ax.set(xlim=(0.5, 10.5), ylim=(0, data.max() * 1.1))

ax.yaxis.set_major_locator(tck.MaxNLocator(integer=True))

ax.bar(short_course_names, means, yerr=np.abs(mean_bounds_list - means))

ax.tick_params(axis='x', labelrotation=0, size=2)

with open(f"Graphs/all scores.png", "wb") as file:
    plt.savefig(file)

fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

ax.tick_params(which="major", length=5, width=1.3)

ax.set_title(f"Teaching Speed", loc="left")
ax.set_xlabel("Course")
ax.set_ylabel("Rating")

# ax.set(xlim=(0.5, 10.5), ylim=(0, data.max() * 1.1))

# ax.yaxis.set_major_locator(tck.MaxNLocator(integer=True))

ax.bar(short_course_names, too_slows, yerr=np.abs(too_slow_bounds_list - too_slows), label="Too Slow")
ax.bar(short_course_names, (100-too_fasts-too_slows), yerr=np.abs(too_fast_bounds_list - too_fasts),
       bottom=too_slows, label="Right Speed")
ax.bar(short_course_names, too_fasts, bottom=(100-too_fasts), label="Too Fast")

ax.tick_params(axis='x', labelrotation=0, size=2)

ax.set(ylim=(0, 100), xlim=(-0.5, 6.5))

ax.legend(loc="upper right", frameon=True)

ax.yaxis.set_major_formatter(tck.PercentFormatter())

with open(f"Graphs/speeds.png", "wb") as file:
    plt.savefig(file)
