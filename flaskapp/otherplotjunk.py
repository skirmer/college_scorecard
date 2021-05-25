
def plot_dist(df, prediction=1):
    vline = hv.VLine(prediction)
    dist = hv.Distribution(list(df['EARN_MDN_HI_2YR'].astype(int)))
    layout = (dist * vline)

    layout.opts(
        opts.Distribution(height = 200, width = 800, tools=['hover']),
        opts(xlabel='Earnings, All Majors'),
        opts(xformatter='$%.0f'),
        opts(yaxis='bare'),
        opts.VLine(color='red', line_width=2)
    )
    return layout

def plot_hist(df, prediction=1):
    hist, edges = np.histogram(df['EARN_MDN_HI_2YR'].astype(int), bins = 40)
    hist_df = pd.DataFrame({column: hist,
                            "left": edges[:-1],
                            "right": edges[1:]})
    hist_df["interval"] = ["%d to %d" % (left, right) for left, 
                            right in zip(hist_df["left"], hist_df["right"])]

    src = ColumnDataSource(hist_df)
    vline = Span(location=prediction, dimension='height', line_color='red', line_width=3)

    plot = figure(plot_height = 600, plot_width = 600,
      title = "Histogram of {}".format(column.capitalize()),
      x_axis_label = column.capitalize(),
      y_axis_label = "Count")  

    plot.quad(bottom = 0, top = column,left = "left", 
        right = "right", source = src, fill_color = colors[0], 
        line_color = "black", fill_alpha = 0.7,
        hover_fill_alpha = 1.0, hover_fill_color = colors[1])
    plot.renderers.extend([vline])

    return plot
    
def plot_groupdist(df, prediction=1, majorfield = 'Accounting and Related Services.'):  
    vline = hv.VLine(prediction)
    dist = hv.Distribution(list(df[df.CIPDESC_new == majorfield]['EARN_MDN_HI_2YR'].astype(int)))
    layout = (dist * vline)

    layout.opts(
        opts.Distribution(height = 200, width = 800, tools=['hover']),
        opts(xlabel=f'Earnings, {majorfield}'),
        opts(xformatter='$%.0f'),
        opts(yaxis='bare'),
        opts.VLine(color='red', line_width=2)
    )
    return layout