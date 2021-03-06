"""
Hash similar documents to same buckets to identify similar documents
"""

def band_hashing(band, hash_f, buckets_dict):
    """helper-function: To Perform hash on bands

    This function takes a band as input,
    hashes it and puts it in the buckets_list
    at its respective postition.
    """

    for col in band.columns:
        h = hash_f(tuple(band[col].values))
        if h in buckets_dict: 
            buckets_dict[h].append(col)
        else: 
            buckets_dict[h] = [col]


def get_bucket_list(sign_mat, r, hash_f=None):
    """This function returns the list of buckets with similar documents

    This function generates the list of dictionaries objects where
    each each band is hashed to a bucket in a dictionary.
    
    Parameters
    ----------
    sign_mat: pandas.Dataframe
        signatures of all the dacuments generated from minhashing
    r: int
        no of rows in each band
    hash_f: function, optional
        hash function used to hash document to buckets

    Returns
    -------
    buckets_list: a list of dictionaries. Each dictionary 
        contains hashes of column vectors of the band as keys 
        and the list of documents as values.
    """

    # b: number of bands
    # n: length of a document signature
    # r: number of rows in a band
    n = sign_mat.shape[0]
    b = n//r
    buckets_list = [dict() for i in range(b)]

    if hash_f==None:
        hash_f = hash

    for i in range(0, n-r+1, r):
        band = sign_mat.loc[i:i+r-1,:]
        band_hashing(band, hash_f, buckets_list[int(i/r)])

    return buckets_list


def query_band_hashing(band, hash_f):
    """helper-function: To Perform hash on query doc bands

    This function takes a band of query document as input,
    hashes it and puts it in the buckets_list
    at its respective postition.
    """

    hash_list = []
    h = hash_f(tuple(band.values))
    hash_list.append(h)
    
    return hash_list


def find_similar_docs(doc_id, buckets_list, sign_mat, r, hash_f=None):
    """This function finds similar documents

    Parameters
    ----------
    buckets_list: list
        list of dictionary objects generated by get_bucket_list
    hash_f: function, optional
        the same hash function used for get_bucket_list
    
    Returns
    -------
    set
        set containing similar documents to given document
    """
    
    # b: number of bands
    # n: length of a document signature
    # r: number of rows in a band
    n = sign_mat.shape[0]
    b = n//r

    if hash_f==None:
        hash_f = hash
    
    query_bucket_list = []

    for i in range(0, n-r+1, r):
        band = sign_mat.loc[i:i+r-1, int(doc_id)]
        query_bucket_list.append(query_band_hashing(band, hash_f))
    
    similar_docs = set()
    for i in range(len(query_bucket_list)):
        for j in range(len(query_bucket_list[i])):
            similar_docs.update(set(buckets_list[i][query_bucket_list[i][j]]))

    return similar_docs


if __name__=='__main__':
    from minhashing import minhash
    from shingling import main
    data = main()
    sign_mat = minhash(data)
