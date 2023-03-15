function from_df_to_X(df::DataFrame, X_columns::Array, X_tickers::Union{Array, Nothing})
    if x_tickers === nothing 
        n_tickers = length(unique(DF[!, "ticker"]))
        dataframe = df[!, X_columns]
        matrix = Matrix(dataframe)
        array = reshape(matrix, (n_tickers, 1, length(X_columns), n_days))
        return Float32.(array)
    else
        n_tickers = length(X_tickers)
        dataframe = filter(row -> row.ticker ∈ X_tickers, df)
        dataframe = dataframe[!, X_columns]
        matrix = Matrix(dataframe)
        array = reshape(matrix, (n_tickers, 1, length(X_columns), n_days))
        return Float32.(array)
    end
end

function from_df_to_X(df::DataFrame)
    matrix = Matrix(df)
    array = reshape(matrix, (n_tickers, 1, length(X_columns), n_days))
    return Float32.(array)
end


function from_df_to_y(df::DataFrame, y_columns::Array, y_ticker::String)
    dataframe = filter(row -> row.ticker == y_ticker, df)
    dataframe = dataframe[!, y_columns]
    matrix = Matrix(dataframe)
    array = reshape(matrix, (1, n_days))
    return Float32.(array)
end

function windows_idxs(samples::UnitRange{Int}, windowsize::Int, stride::Int) 
    windows_idxs =  [i+(i-1)*(stride-1):i+(i-1)*(stride-1)+windowsize-1 for i in samples if (i+(i-1)*(stride-1)+windowsize-1) <= samples[end]]
    return windows_idxs
end

function get_windows(array::Array, w_idxs::Array{UnitRange{Int}})
    windowsize = length(w_idxs[1])
    N_windows = length(w_idxs)
    array_windowed = Array{Float32}(undef, size(array)[1:end-1]..., windowsize, N_windows)
    N_colons = repeat([:], ndims(array_windowed)-1)
    for i in eachindex(w_idxs)
        array_windowed[N_colons..., i] = selectdim(array, length(size(array)), w_idxs[i])
    end
    return array_windowed
end

function windowing_batching(array::Array, w_idxs::Array{UnitRange{Int}}, b_idxs::Int)
    array_windows = get_windows(array, w_idxs)
    array_bw = get_windows(array_windows, b_idxs)
    return array_bw
end

function futures_idxs(samples::UnitRange{Int}, windowsize::Int, stride::Int, futurestep::Int, futurews::Int=1) 
    futures_idxs = [i+(i-1)*(stride-1)+windowsize-1+futurestep for i in samples if (i+(i-1)*(stride-1)+windowsize+futurestep-1) <= samples[end]]
    wf_idxs = windows_idxs(futures_idxs, futurews, 1)
    return wf_idxs
end

function windows_futures_idxs(samples::UnitRange{Int}, windowsize::Int, stride::Int, futurestep::Int, futurews::Int=1)
    w_idxs = windows_idxs(samples, windowsize, stride)
    f_idxs = futures_idxs(samples, windowsize, stride, futurestep, futurews)
    w_idxs = length(w_idxs) > length(f_idxs) ? w_idxs[1:length(f_idxs)] : w_idxs
    return w_idxs, f_idxs
end