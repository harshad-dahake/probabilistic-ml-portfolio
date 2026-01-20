"""
Metropolis-Hastings MCMC for Cryptanalysis
------------------------------------------
Author: Harshad Dahake
Context: Gatsby Unit Coursework (Probabilistic and Unsupervised Learning)

Description:
  A Markov Chain Monte Carlo (MCMC) sampler designed to break substitution ciphers.
  The model treats the decryption key as a latent permutation variable and samples 
  from the posterior distribution P(Key | Encrypted Text) using the Metropolis-Hastings algorithm.

  The transition kernel proposes swaps in the cipher mapping, and the acceptance 
  probability is driven by the likelihood of the resulting text statistics (bigram frequencies).

Key Concepts:
  - Markov Chain Monte Carlo (MCMC)
  - Combinatorial Optimization via Sampling
  - Discrete Latent Spaces
  
Note: Requires 'symbols.txt' and a reference corpus (e.g., 'War and Peace') to execute.
"""

# %% Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # %% ## Read symbols into symbols table

    # read file contents
    with open("symbols.txt") as f:
        sym = f.read()
        # print("Here're the symbols:\n")
        # print(sym)

    symbols_table = pd.DataFrame(columns=["symbols"])

    # process imported file data (while considering line-break character (1 per line)
    z = -1
    for i,s in enumerate(sym):
        z = z*(-1)
        if z == 1:
            # print(f"{(i//2)}: {s}")
            symbols_table.loc[(i//2), "symbols"] = s
    del z, sym, f

    # %% ## Import training message data: War and Peace | Pre-process data

    # import Training Message
    with open("2600-0.txt") as f:
        wap = f.read()

    ## NOTE: FIGURED OUT THE START AND END OF THE ACTUAL BOOK TEXT FROM THE .txt FILE
    # Used trial and error method
    fbc = 7485 # first_book_char (first legit book text character)
    lbc = 3209107 # last_book_char (last legit book text character)

    # get the main textbook contents
    wap_main = wap[fbc:lbc]


    # %% ## Pre-process (clean) data to match the symbols table as well as standardize special characters and symbols

    # Clear representations with multiple spaces, new paragraphs or chapters, etc.
    wap_main_clnd = ' '.join(wap_main.split())

    import unicodedata
    import string

    def clean_text_for_char_counting(raw_text):
        # Consolidate Quotation Marks and Hyphes
        
        # Variations of ' " '
        quotation_translation_table = str.maketrans('“”', '""')

        single_quote_translation_table = str.maketrans('‘’', "''")

        # Variations of ' - '
        dash_translation_table = str.maketrans('—', '-')

        ligature_map = str.maketrans({
            'œ': 'oe',
            'æ': 'ae'
        })

        # Combine the translation tables
        combined_translation_table = str.maketrans({**quotation_translation_table, **single_quote_translation_table, **dash_translation_table, **ligature_map})
        
        cleaned_text = raw_text.translate(combined_translation_table)

        ## Remove Accents/Diacritics (Variants of a, u, i, etc.)
        
        normalized_text = unicodedata.normalize('NFKD', cleaned_text)
        text_without_diacritics = "".join(
            [c for c in normalized_text if not unicodedata.combining(c)]
        )
        
        return text_without_diacritics

    # call the clean-up function to process the book text
    wap_main_clnd_std = clean_text_for_char_counting(wap_main_clnd)

    # convert full text to lower case so that 'A' and 'a' count as the same thing, for example
    wap_main_clnd_std = wap_main_clnd_std.lower()
    # %% ## CALCULATE STATIONARY DISTRIBUTION ESTIMATES (SMOOTHED) FOR TRAINING TEXT (WAR AND PEACE)

    ## GET FREQUENCES (SINGLE CHARACTER) IN A DICTIONARY ##
    from collections import Counter

    # get count of single characters
    wap_syms1 = dict(Counter(wap_main_clnd_std))

    # arrange in decreasing order (helps with identifying a potentially decent starting point for MCMC later on)
    desc_wap_syms1 = dict(sorted(wap_syms1.items(), key=lambda item: item[1], reverse=True))

    # calculate total number of characters
    total_chars = sum(desc_wap_syms1.values())

    # calculate frequency proportions
    wap_syms1_perc = {char: np.round(count / total_chars, 4) for char, count in desc_wap_syms1.items()}


    ## GET FREQUENCES (SINGLE CHARACTER) INTO A DATA FRAME AND CALCULATE STATIONARY DISTRIBUTION ##
    sym_counts1 = symbols_table.copy()
    sym_counts1["frequency"] = np.zeros(len(sym_counts1))

    i = 0
    for sym in sym_counts1["symbols"]:
        # print(f"{sym}")
        if desc_wap_syms1.get(sym) != None:
            sym_counts1.loc[i, "frequency"] = desc_wap_syms1.get(sym)
        i += 1

    sym_counts1.sort_values(by='frequency', ascending=False, inplace=True)

    seq_len = len(wap_main_clnd_std) # print(f"msg length: {seq_len}")
    sym_counts1["stat_dist"] = sym_counts1["frequency"] / seq_len


    ## SMOOTHEN THE UNIGRAM FREQUENCY VECTOR TO ALLOW FOR TWO MISSING CHARACTERS FROM TRAINING TEXT: "[" AND "]" ##
    alpha = 1
    vocab_full = len(symbols_table)

    # Calculate the denominator for smoothing: (Original Total Count) + alpha * V_full
    # Note: (Original total count) is the length of the sequence (seq_len)
    smooth_denominator = seq_len + (alpha * vocab_full)

    # Apply smoothing to the numerator: Frequency (Count) + alpha
    smoothed_frequency = sym_counts1["frequency"] + alpha

    # Calculate the Smoothed Stationary Distribution
    sym_counts1["stat_dist_smoothed"] = smoothed_frequency / smooth_denominator

    # reset index of data frame for later processing
    sym_counts1.reset_index(inplace=True)

    # %% ## C

    ## GET FREQUENCIES (OF BIGRAMS: ORDERED PAIR OF CHARACTERS) IN A MATRIX FOR WAR AND PEACE (TRAINING TEXT)

    bigrams = zip(wap_main_clnd_std[:-1], wap_main_clnd_std[1:])

    # Convert the pairs into strings for easy counting, e.g., ('s', 'e') -> 'se'
    bigram_list = [''.join(pair) for pair in bigrams]

    # Separate the characters into two lists for crosstab
    first_chars = [pair[0] for pair in bigram_list]
    second_chars = [pair[1] for pair in bigram_list]

    # Count the occurrences of each bigram
    bigram_counts = Counter(bigram_list)

    # Calculate total number of bigrams
    total_bigrams = sum(bigram_counts.values())

    # Calculate bigram frequencies
    bigram_frequencies = {bigram: count / total_bigrams for bigram, count in bigram_counts.items()}


    ### TRANSITION PROBABILITIES WITH SMOOTHING TO AVOID NUMERICAL ISSUES DOWNSTREAM ###

    # NOTE: GIVEN THE ORIGINAL TEXT DOESN'T HAVE THE TWO CHARACTERS '[' AND ']', WE NEED TO UPDATE THE OBTAINED BIGRAM COUNT MATRIX FROM 51X51 TO 53X53

    ## 1. Create the raw count matrix with
    # rows: first character of bigram
    # cols: second character of bigram
    count_matrix = pd.crosstab(index=first_chars, columns=second_chars, dropna=False)

    # Get list of all 53 symbols
    full_vocab = sym_counts1["symbols"].tolist()

    # Create placeholder dataframe/matrix with 53x53 shape
    full_count_matrix = pd.DataFrame(0, index=full_vocab, columns=full_vocab)

    # Integrate counts from the 51x51 matrix into the new matrix
    full_count_matrix = full_count_matrix.add(count_matrix, fill_value=0)


    ## Using Laplace Smoothing Technique (adding alpha to every cell)
    V = full_count_matrix.shape[1] # total number of symbols (53)
    alpha = 1 # Set the smoothing parameter (alpha = 1 for Laplace)

    # Update numerator for each cell update (add alpha)
    smoothed_counts = full_count_matrix + alpha

    # Update the denominator for each cell update (Original Row Sum + alpha * V)
    original_row_sums = full_count_matrix.sum(axis=1)

    # Add smoothing correction to denominator
    new_denominator = original_row_sums + alpha * V

    # Calculate the smoothed transition probability matrix
    smoothed_frequency_matrix = smoothed_counts.divide(new_denominator, axis='index')

    # Update frequency_matrix to this new "full" and "smoothed" matrix
    frequency_matrix = smoothed_frequency_matrix

    ### NOTE: frequency_matrix is the transition probability matrix ###

    # %% ## IMPORT ENCRYPTED MESSAGE ##

    with open("message.txt") as f:
        msg = f.read()
        print("Here's the message:\n")
        print(msg)

    # %% ## Process encrypted message

    # Gather all symbols and corresponding frequencies for encrypted message
    symbols = {}
    for char in msg:
        # print(char)
        if char in symbols.keys():
            symbols[char] += 1
        else:
            symbols[char] = 1

    # arrange in decreasing order (helps with identifying a potentially decent starting point for MCMC later on)
    desc_symbols = dict(sorted(symbols.items(), key=lambda item: item[1], reverse=True))

    # Convert to data frame
    sym_counts = symbols_table.copy()
    sym_counts["frequency"] = np.zeros(len(sym_counts))

    i = 0
    for sym in sym_counts["symbols"]:
        if symbols.get(sym) != None:
            sym_counts.loc[i, "frequency"] = symbols.get(sym)
        i += 1
    sym_counts.sort_values(by='frequency', ascending=False, inplace=True)

    # Get 'stationary' distribution of characters in encrypted message (helps with identifying a potentially decent starting point for MCMC later on)
    seq_len = len(msg)
    sym_counts["stat_dist"] = sym_counts["frequency"] / seq_len

    # reset index
    sym_counts.reset_index(inplace=True)

    # %% ## Define Metropolis-Hastings Sampler

    ### MH SAMPLER ###

    ## start with an initial key: sigma_initial

    # define the random variable sigma
    sigma = pd.DataFrame()

    # initialise sigma to sigma_init
    sigma["char"] = sym_counts1["symbols"]
    sigma["key"] = sym_counts["symbols"]
    # sigma

    sigma_init = sigma.copy()
    # sigma_init


    # %% ## MARKOV CHAIN MONTE CARLO SAMPLING ALGORITHM

    sigma_curr = sigma_init.copy() # if starting afresh
    # sigma = sigma_curr.copy() # if continuing from previous iterations
    l = 10000
    a = np.zeros(l)
    b = np.zeros(l)
    log_lh = np.zeros(l)
    chain_history_sigmas = []

    iters = 1 # keep count of total iterations up to convergence or termination
    log_lh_max = 0

    for j in range(l):
        
        # periodically print the current max log likelihood along with the latest log likelihood (corresponding to the currently selected sigma)
        if j%50 == 0:
            print(f"j = {j}")
            print(f"Current log likelihood max: {log_lh_max}, with latest log likelihood: {log_lh[j-1]}\n")
        
        # for the first iteration, evaluate log-likelihood at initial sigma
        if j == 0:
            log_lh[j] = 0
            # for each character e in the encrypted message msg, process log likelihood based on stationary distribution and transition probabilities estimates
            for i,e in enumerate(msg):
                if i == 0:
                    log_lh[j] = np.log(sym_counts1[sym_counts1["symbols"] == sigma[sigma["key"] == e]["char"].item()]["stat_dist_smoothed"].item()) # first character
                if i > 0:
                    prev = sym_counts1[sym_counts1["symbols"] == sigma[sigma["key"] == msg[i-1]]["char"].item()]["symbols"].item() # previous char
                    curr = sym_counts1[sym_counts1["symbols"] == sigma[sigma["key"] == e]["char"].item()]["symbols"].item() # current char
                    log_lh[j] += np.log(frequency_matrix.loc[prev,curr].item()) # transition probability from prev char (msg[i-1]) to curr char (msg[i])
            
            # update the max total log likelihood observed till now to the calculated log-likelihood
            log_lh_max = log_lh[j]
        
        # MCMC sampling algorithm
        if j > 0:
            ## Propose a new sigma
            # select two indices at random
            a[j] = np.floor(np.random.uniform(low=0,high=1)*53)
            b[j] = np.floor(np.random.uniform(low=0,high=1)*53)
            # ensure we select two different indices
            while b[j] == a[j]:
                b[j] = np.floor(np.random.uniform(low=0,high=1)*53)
            
            ## Swap indices' mapping to get new key: sigma'
            sigma = sigma_curr.copy() # update sigma to current key (we always evaluate on 'sigma' variable)

            # Swap the corresponding decrypted (Enlish text) characters ('char' column) at the two random indices
            
            # Get the values for characters at index a and b
            char_a = sigma.loc[a[j], 'char']
            char_b = sigma.loc[b[j], 'char']

            # Swap the char values
            sigma.loc[a[j], 'char'] = char_b
            sigma.loc[b[j], 'char'] = char_a

            # evaluate p(sigma') - LOG LIKELIHOOD FORMAT
            log_lh[j] = 0
            for i,e in enumerate(msg):
                if i == 0:
                    log_lh[j] = np.log(sym_counts1[sym_counts1["symbols"] == sigma[sigma["key"] == e]["char"].item()]["stat_dist_smoothed"].item()) # first char
                if i > 0:
                    prev = sym_counts1[sym_counts1["symbols"] == sigma[sigma["key"] == msg[i-1]]["char"].item()]["symbols"].item() # prev char
                    curr = sym_counts1[sym_counts1["symbols"] == sigma[sigma["key"] == e]["char"].item()]["symbols"].item() # curr char
                    log_lh[j] += np.log(frequency_matrix.loc[prev,curr].item()) # transition probability from msg[i-1] to msg[i]

            # update log_lh_max 'till date' if new log likelihood is greater than the currently known max value
            if log_lh[j] > log_lh_max:
                log_lh_max = log_lh[j]
            
            # evaluate log(p(sigma')) - log(p(sigma)) as diff_plogs, compare with 0
            diff_plogs = log_lh[j] - log_lh[j-1]

            # if diff_plogs > 0, accept with p = 1
            # else accept with p = exp(diff_plogs) --- sample alpha randomly from uniform distribution on [0,1], accept if diff_plogs > alpha
            if diff_plogs > 0:
                accept = True
            else:
                alpha = np.exp(diff_plogs)
                beta = np.random.uniform(low=0, high=1)

                if beta > alpha:
                    accept = False

                else:
                    accept = True
            
            # If sigma' is accepted, update sigma_curr to sigma'. Otherwise, we reject => we stay at the current sigma.
            # Therefore, update log likelihood after this sample to the previous step's log likelihood
            if accept == True:
                sigma_curr = sigma.copy()
            else:
                log_lh[j] = log_lh[j-1]
            
        ## Decrypt first 60 characters of the encrypted message using current key (sigma_curr) after every 100 MCMC samples/iterations
        if (j+1)%100 == 0:
            decrypted = []
            for e in msg[:60]:
                s = sigma_curr[sigma_curr["key"]==e]["char"].item()
                decrypted.append(s)
            # print(f"decrypted: {decrypted}")
            decrypted_msg = ''.join(decrypted)
            
            print(f"Decrypted message #{(j//100) + 1} - first 60 characters:\n{decrypted_msg}\n")
        
        ## Update history of accepted sigma (keys)
        chain_history_sigmas.append(sigma_curr)
        iters += 1


    # %% ## Define Metropolis-Hastings Sampler

    # Plot Log-likelihood vs Iterations to study progress and check for convergence
    plt.figure(figsize=(12, 8)) # Set size for better visibility
    plt.plot(range(1,iters), log_lh[:iters], label='Log-Likelihood per Iteration', color='blue', alpha=0.8)
    plt.title('MCMC Chain Convergence (Log-Likelihood)')
    plt.xlabel('Iteration Number')
    plt.ylabel('Log-Likelihood')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


    # %% ## DECRYPT WHOLE MESSAGE WITH FINAL KEY

    # Decrypt message with final key
    decrypted = []
    for e in msg:
        s = sigma_curr[sigma_curr["key"]==e]["char"].item()
        decrypted.append(s)
    # print(f"decrypted: {decrypted}")
    decrypted_msg = ''.join(decrypted)

    print(f"Decrypted message:\n{decrypted_msg}\n")


    # %% ## Fetch the final decryption key
    sigma_curr

    # %% ## Fetch the stationary distribution estimates
    sym_counts1

    # %% ## Fetch the transition probabilities
    frequency_matrix


# %% Main function executor
if __name__ == "__main__":
    main()