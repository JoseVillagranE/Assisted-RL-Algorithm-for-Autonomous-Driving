import numpy as np
import matplotlib.pyplot as plt

class Dynamic2DFigure():
    def __init__(self,
                 figsize=(8,8),
                 edgecolor="black",
                 rect=[0.1, 0.1, 0.8, 0.8],
                 ylim=[0, 100],
                 *args, **kwargs):
        
        self.ymin = ylim[0]
        self.ymax = ylim[1]
        
        self.graphs = {}
        self.texts = {}
        self.fig = plt.figure(figsize=figsize, edgecolor=edgecolor)
        self.ax = self.fig.add_axes(rect)
        self.ax.set_ylim(ylim[0], ylim[1])
        # self.fig.tight_layout()
        self.marker_text_offset = 0
        if kwargs["title"] is not None:
            self.fig.suptitle(kwargs["title"])
        self.axis_equal = False
        self.invert_xaxis = False
        self.invert_yaxis = False

    def set_invert_x_axis(self):
        self.invert_xaxis = True
        
    def set_invert_y_axis(self):
        self.invert_yaxis = True

    def set_axis_equal(self):
        self.axis_equal = True

    def add_graph(self, name, label="", window_size=10, x0=None, y0=None,
                  linestyle='-', linewidth=1, marker="", color="k",
                  markertext=None, marker_text_offset=2):
        
        self.marker_text_offset = marker_text_offset

        if x0 is None or y0 is None:
            x0 = np.zeros(window_size)
            y0 = np.zeros(window_size)
            new_graph, = self.ax.plot(x0, y0, label=label,
                                      linestyle=linestyle, linewidth=linewidth,
                                      marker=marker, color=color)
            if markertext is not None:
                new_text = self.ax.text(x0[-1], y0[-1] + marker_text_offset,
                                         markertext)
        else:
            new_graph, = self.ax.plot(x0, y0, label=label,
                                      linestyle=linestyle, linewidth=linewidth,
                                      marker=marker, color=color)
            if markertext is not None:
                new_text = self.ax.text(x0[-1], y0[-1] + marker_text_offset,
                                         markertext)

        self.graphs[name]           = new_graph
        if markertext is not None:
            self.texts[name + "_TEXT"] = new_text

    def roll(self, name, new_x, new_y):
        graph = self.graphs[name]
        if graph is not None:
            x, y = graph.get_data()
            x = np.roll(x, -1)
            x[-1] = new_x
            y = np.roll(y, -1)
            y[-1] = new_y
            graph.set_data((x, y))
            self.rescale()
        if name + "_TEXT" in self.texts:
            graph_text = self.texts[name + "_TEXT"]
            x = new_x
            y = new_y + self.marker_text_offset
            graph_text.set_position((x, y))
            self.rescale()
# 
    def update(self, name, new_x_vec, new_y_vec, new_colour='k'):
        graph = self.graphs[name]
        if graph is not None:
            graph.set_data((np.array(new_x_vec), np.array(new_y_vec)))
            graph.set_color(new_colour)
            self.rescale()
        if name + "_TEXT" in self.texts:
            graph_text = self.texts[name + "_TEXT"]
            x = new_x_vec[-1]
            y = new_y_vec[-1] + self.marker_text_offset
            graph_text.set_position((x, y))
            self.rescale()

    def rescale(self):
        xmin = float("inf")
        xmax = -1*float("inf")
        ymin, ymax = self.ax.get_ylim()
        for name, graph in self.graphs.items():
            xvals, yvals = graph.get_data()
            xmin_data = xvals.min()
            xmax_data = xvals.max()
            ymin_data = yvals.min()
            ymax_data = yvals.max()
            xmin_padded = xmin_data-0.05*(xmax_data-xmin_data)
            xmax_padded = xmax_data+0.05*(xmax_data-xmin_data)
            ymin_padded = ymin_data-0.05*(ymax_data-ymin_data)
            ymax_padded = ymax_data+0.05*(ymax_data-ymin_data)
            xmin = min(xmin_padded, xmin)
            xmax = max(xmax_padded, xmax)
            ymin = min(ymin_padded, ymin)
            ymax = max(ymax_padded, ymax)
        self.ax.set_xlim(xmin, xmax)
        # self.ax.set_ylim(ymin, ymax)
        self.ax.set_ylim(self.ymin, self.ymax)
        if self.axis_equal:
            self.ax.set_aspect('equal')
        if self.invert_xaxis:
            self.ax.invert_xaxis()
        if self.invert_yaxis:
            self.ax.invert_yaxis()


class LivePlotter2():
    
    def __init__(self):
        self._default_w = 150
        self._default_h = 100
        self._graph_w = 0
        self._graph_h = 0
        self._figs = []
    
    def refresh_figure(self, fig):
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    def refresh(self):
        for fig in list(self._figs):
            self.refresh_figure(fig)
        
    def plot_figure(self, fig):

        f_w = fig.get_window_extent().width
        f_h = fig.get_window_extent().height
        f_w, f_h = int(f_w), int(f_h)
        self._graph_h += f_h
        self._graph_w = max(self._graph_w, f_w)

        self._figs.append(fig)
        
    def plot_new_dynamic_2d_figure(self, title="", **kwargs):
        dy2dfig = Dynamic2DFigure(title=title, **kwargs)
        fig = dy2dfig.fig
        # this stores the figure locally as well
        self.plot_figure(fig)
        return dy2dfig
    
    
    
    
if __name__ == "__main__":
    
    from scipy.interpolate import interp1d
    
    lp = LivePlotter2()
    
    fig = lp.plot_new_dynamic_2d_figure(
                    title='Test Trajectory',
                    figsize=(6, 6),
                    edgecolor="black",
                    rect=[.1, .1, .8, .8],
                    ylim=[0, 1])
    
    fig.set_invert_y_axis()
    plt.show(block=False)
    
    num_points = 300
    INTERP_MAX_POINTS_PLOT = 20
    OFFSET = 0.1
        
    x = np.arange(num_points)
    y = np.array([0.5]*num_points)
    x0 = x[0]
    y0 = 0.5
    # y0 = np.sin(x0)
    
    fig.add_graph("waypoints", window_size=num_points, x0=x, y0=y,
                  linestyle="-", marker="", color='g')
    
    fig.add_graph("agent", window_size=num_points, x0=[x0]*num_points,
                      y0=[y0]*num_points,
                      linestyle="-", marker="", color=[1, 0.5, 0])
    
    fig.add_graph("start_pos", window_size=1, 
                                  x0=[x0], y0=[y0],
                                  marker=11, color=[1, 0.5, 0], 
                                  markertext="Start", marker_text_offset=1)
    # Add end position marker
    fig.add_graph("end_pos", window_size=1, 
                              x0=[x[-1]], 
                              y0=[y[-1]],
                              marker="D", color='r', 
                              markertext="End", marker_text_offset=1)
    # Add car marker
    fig.add_graph("car", window_size=1, 
                              marker="s", color='b', markertext="Car",
                              marker_text_offset=1)

    # Add local path proposals
    for i in range(5):
        fig.add_graph("local_path " + str(i), window_size=num_points,
                                  x0=None, y0=None, color=[0.0, 0.0, 1.0])
    
    for i in range(num_points):
        cx = x[i]
        cy = y[i]
        look_ahead_points = INTERP_MAX_POINTS_PLOT
        if i + INTERP_MAX_POINTS_PLOT < num_points:
            cx_1 = x[i+INTERP_MAX_POINTS_PLOT]
            cy_1 = y[i+INTERP_MAX_POINTS_PLOT]
            
        else:
            cx_1 = x[-1]
            cy_1 = y[-1]
            look_ahead_points = num_points - i
        
        cy_11 = cy_1 + OFFSET*np.sin(0)
        cy_12 = cy_1 + OFFSET*np.sin(np.pi/4)
        cy_13 = cy_1 + OFFSET*np.sin(np.pi/2)
        cy_14 = cy_1 + OFFSET*np.sin(-np.pi/4)
        cy_15 = cy_1 + OFFSET*np.sin(-np.pi/2)
        
        f = []  
        f.append(interp1d([cx, cx_1], [cy, cy_11]))
        f.append(interp1d([cx, cx_1], [cy, cy_12]))
        f.append(interp1d([cx, cx_1], [cy, cy_13]))
        f.append(interp1d([cx, cx_1], [cy, cy_14]))
        f.append(interp1d([cx, cx_1], [cy, cy_15]))
        
        fig.roll("agent", cx, cy)
        fig.roll("car", cx, cy)
        for j in range(5):
            fig.update("local_path " + str(j), x[i:i+look_ahead_points],
                                  f[j](x[i:i+look_ahead_points]), "blue")
        
        lp.refresh()
        # live_plot_timer.lap()

