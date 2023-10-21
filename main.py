from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv', sep=';', encoding='mbcs')

# Traitement de base df
df = df.dropna(subset=['Avg_Daily_Visitors'])
df = df.drop_duplicates(subset=['Website'])
df['website_name'] = df['Website'].str.split('.').str[1]

# Couleur en fonction du child_safety
dict_color_par_cat = {
    'Excellent':'#47D64E',
    'Good':'#B2EA24',
    'Poor':'#DCA500',
    'Unsatisfactory':'#ff6600',
    'Very poor':'#ff0000',
    'Unknown':'#9A9A9A',
}

df['color'] = df['Child_Safety'].map(dict_color_par_cat)

# Réduire dataframe avec les x plus gros sites
def parse_dataframe(df,lendf):
    df_parse = df.sort_values(by=['Avg_Daily_Visitors'], ascending=False)[:lendf]
    df_parse = df_parse.reset_index(drop=True)
    return df_parse

# Dataframe avec les x plus gros site
df_max_20 = parse_dataframe(df, 20)
df_max_50 = parse_dataframe(df, 50)
df_max_200 = parse_dataframe(df, 200)




class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return np.argmin(distance, keepdims=True)

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        """

        min_size = 5
        max_size = 20
        normalized_sizes = np.interp(self.bubbles[:, 2], (self.bubbles[:, 2].min(), self.bubbles[:, 2].max()), (min_size, max_size))

        for i in range(len(self.bubbles)):
            circ = plt.Circle(self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)

            fontsize = int(normalized_sizes[i])

            ax.text(*self.bubbles[i, :2], labels[i], horizontalalignment='center', verticalalignment='center',fontsize=fontsize)




# Générer graphique bubble chart
def gen_graph(df, value, name, color, title, output):
    bubble_chart = BubbleChart(area=df[value], bubble_spacing=100)

    bubble_chart.collapse()

    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
    bubble_chart.plot(ax, df[name], df[color])
    ax.axis("off")
    ax.relim()
    ax.autoscale_view()
    ax.set_title(title)

    # Légende personnalisé en fonction du child_safety et donc de la couleur
    legend_elements = [Patch(facecolor='#47D64E', edgecolor='black', label='Excellent'),
                       Patch(facecolor='#B2EA24', edgecolor='black', label='Bon'),
                       Patch(facecolor='#DCA500', edgecolor='black', label='Mauvais'),
                       Patch(facecolor='#ff6600', edgecolor='black', label='Insatisfaisant'),
                       Patch(facecolor='#ff0000', edgecolor='black', label='Tres mauvais'),
                       Patch(facecolor='#9A9A9A', edgecolor='black', label='Inconnu')]

    # Axe de la légende
    legend_ax = fig.add_axes([0.83, 0, 0.1, 0.6])
    legend_ax.axis('off')

    legend = legend_ax.legend(handles=legend_elements, loc='center left', fontsize=7)
    # legend.set_title("Child safety rank", prop={'size': 7})
    legend.set_title("Adapté aux enfants", prop={'size': 7})

    # Enregistrement
    plt.savefig(output, dpi=300)

    plt.close()

    print('graphique : "' + title + '" généré.')


# gen 25 graphs website aléatoires
for i in range(0,25):
    # Prendre 10 site aléatoirement
    df_sample = df.sample(n=10)
    df_sample = df_sample.reset_index(drop=True)

    gen_graph(df_sample, 'Avg_Daily_Visitors', 'website_name', 'color', 'Visualisation de la popularité de 10 sites web aléatoires', 'output/website'+str(i)+'.png')


# Full 20 +
gen_graph(df_max_20, 'Avg_Daily_Visitors', 'website_name', 'color', 'Visualisation de la pupularité des 20 sites web les plus visités', 'output/full20+.png')

# Full 50 +
gen_graph(df_max_50, 'Avg_Daily_Visitors', 'website_name', 'color', 'Visualisation de la pupularité des 50 sites web les plus visités', 'output/full50+.png')

# Full 200 +
gen_graph(df_max_200, 'Avg_Daily_Visitors', 'website_name', 'color', 'Visualisation de la pupularité des 200 sites web les plus visités', 'output/full200+.png')
