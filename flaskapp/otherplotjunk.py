
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

def plot_dist(df=major, prediction=55000):
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
    
def plot_groupdist(df=major, prediction=55000, majorfield = 'Accounting and Related Services.'):  
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