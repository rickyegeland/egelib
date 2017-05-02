import numpy as np
import astropy
import astroquery.simbad
import astroquery.vizier
import egelib.timeseries
import egelib.stellarprop

def query_simbad(obj):
    simbad = astroquery.simbad.Simbad()
    simbad.remove_votable_fields('coordinates')
    simbad.add_votable_fields('sp', 'flux(B)', 'flux(V)')
    result = simbad.query_object(obj)
    result['FLUX_B-V'] = result['FLUX_B'] - result['FLUX_V']
    return result

def query_mes(obj, mes):
    simbad = astroquery.simbad.Simbad()
    request = "\n".join(['format object f1 "%%MEASLIST(%s)"' % mes, 'id %s' % obj])
    request = dict(script=request)
    result = astroquery.utils.commons.send_request(simbad.SIMBAD_URL, request, simbad.TIMEOUT)
    result = astroquery.simbad.core.SimbadResult(result.text)
    return result.data

def query_feh(obj):
    result = query_mes(obj, 'fe_h')
    if result is None:
        raise Exception("No FeH data in SIMBAD for object "+obj)
    table = astropy.io.ascii.read(result,
                                  format='fixed_width',
                                  names=('mes', 'Teff', 'logg', 'Fe_H', 'flag', 'CompStar', 'CatNo', 'References'),
                                  col_starts=(0,  8, 14, 20, 25, 27, 37, 43),
                                  col_ends=  (3, 11, 17, 23, 25, 35, 41, 61))
    return table

def query_feh_median(obj):
    feh = query_feh(obj)
    cols = ('Teff', 'logg', 'Fe_H')
    table = astropy.table.Table([[obj]], names=['ID'])
    for col in cols:
        valid = np.logical_not(feh[col].mask)
        coldata = feh[col][valid]
        table[col+'_med'] = [np.median(coldata)]
        table[col+'_madstd'] = [egelib.stats.mad_sigma(coldata)]
        table[col+'_N'] = [valid.sum()]
    return table

def query_logg(obj, pref=('J/AJ/141/90/table2-1','J/AJ/141/90/table2-2', 'J/AJ/141/90/table1-2')):
    cats = {'J/AJ/141/90/table2-1': ('logg1', 'e_logg1', '2011AJ....141...90L'),
            'J/AJ/141/90/table2-2': ('logg2', 'e_logg2', '2011AJ....141...90L'),
            'J/AJ/141/90/table1-2': ('logg2', 'e_logg2', '2011AJ....141...90L')}
    for cat in pref:
        logg_col, e_col, ref = cats[cat]
        if '-' in cat:
            cat, extra = cat.split('-')
        result = astroquery.vizier.Vizier.query_object(obj, catalog=[cat])
        N = len(result) # number of tables in result
        if N == 0:
            continue
        result = result[0]
        N = len(result) # number of rows in first table
        logg = result[logg_col]
        e_logg = result[e_col]
        # If N > 1, takes average.  If N=1, converts 1-element array to scalar
        logg = np.mean(logg)
        e_logg = np.mean(e_logg) / np.sqrt(N)
        table = astropy.table.Table([[obj], [logg], [e_logg], [ref]], names=('INPUT', 'logg', 'e_logg', 'logg_ref'))
        return table
    raise Exception('No log(g) results found in any catalog')

def query_gcs3(obj):
    # Queries the GCS3
    columns = ['HIP', 'b-y', 'm1', 'c1', 'logTe', '[Fe/H]', 'plx', 'e_plx', 'Dist', 'VMag', 'e_VMag', 'Vmag']
    V = astroquery.vizier.Vizier(catalog=['V/130/gcs3'], columns=columns)
    result = V.query_object(obj, radius=10.*astropy.units.arcsec)
    # check number of tables: expect 1
    if len(result) == 1:
        result = result[0]
    else:
        raise Exception('No GCS3 results for object '+obj)
    
    # check number of rows: expect 1
    if len(result) > 1:
        raise Exception('Multiple (%i) GCS3 rows for object %s' % (len(result), obj))
    elif len(result) == 0:
        raise Exception('No GCS3 rows for object '+obj)
    # astropy Table renames [Fe/H] column; fix it
    result.rename_column('__Fe_H_', 'Fe_H')
    # Replace logTe with Teff
    result['logTe'] = 10**result['logTe']
    result.rename_column('logTe', 'Teff')
    result['e_Teff'] = 57. # from GCS3 paper
    # Vizier Query adds coordinate columns; remove them
    result.remove_column('_RAJ2000')
    result.remove_column('_DEJ2000')
    # Vizier query reorders columns; fix it
    columns[4] = 'Teff'
    columns[5] = 'Fe_H'
    columns.insert(5, 'e_Teff')
    return result[columns]

def nulltable(dtype):
    nt = astropy.table.Table(dtype=dtype, masked=True)
    nt.add_row()
    for c in nt.colnames:
        nt[c].mask = True
    return nt

def query_TLRM(obj):
    """Query SIMBAD and GCS to determine temperature, luminosity, radius, and mass"""
    simbad = query_simbad(obj)
    try:
        gcs = query_gcs3(obj)
    except Exception, e:
        print "WARNING: query_gcs3() failed:", e
        # Programing pitfall: updating the columns above will require
        # an update to these dtype specifications.
        gcs = nulltable([('HIP', '<i4'), ('b-y', '<f4'), ('m1', '<f4'), ('c1', '<f4'),
                         ('Teff', '<f4'), ('e_Teff', '<f8'), ('Fe_H', '<f4'),
                         ('plx', '<f4'), ('e_plx', '<f4'), ('Dist', '<i2'),
                         ('VMag', '<f4'), ('e_VMag', '<f4'), ('Vmag', '<f4')])
    try:
        med = query_feh_median(obj)
    except Exception, e:
        print "WARNING: query_feh_median() failed:", e
        med = nulltable([('ID', 'S7'), ('Teff_med', '<f8'), ('Teff_madstd', '<f8'), ('Teff_N', '<i8'),
                         ('logg_med', '<f8'), ('logg_madstd', '<f8'), ('logg_N', '<i8'),
                         ('Fe_H_med', '<f8'), ('Fe_H_madstd', '<f8'), ('Fe_H_N', '<i8')])
    try:
        logg = query_logg(obj)
    except Exception, e:
        print "WARNING: query_logg() failed:", e
        logg = nulltable([('INPUT', 'a8'), ('logg', 'f'), ('e_logg', 'f'), ('logg_ref', 'a19')])
    table = astropy.table.hstack([simbad, gcs, med, logg])
    # Fundemental Properties
    T, e_T = egelib.stellarprop.Teff_to_T(table['Teff'], table['e_Teff'])
    L, R, e_L, e_R = egelib.stellarprop.VMagTeff_to_LR(table['VMag'], table['Teff'], table['e_VMag'], table['e_Teff'])
    T.unit = e_T.unit = L.unit = e_L.unit = R.unit = e_R.unit = None; 
    M, e_M = egelib.stellarprop.mass(table['logg'], R, table['e_logg'], e_R)
    table['T'] = T
    table['e_T'] = e_T
    table['L'] = L
    table['e_L'] = e_L
    table['R'] = R
    table['e_R'] = e_R
    table['M'] = M
    table['e_M'] = e_M
    return table

def query_TLRM_list(obj_list):
    tables = []
    for obj in obj_list:
        print "===", obj
        try:
            table = query_TLRM(obj)
            Nrows = len(table)
            if Nrows > 1:
                print "WARNING: %i rows from query; taking only first row" % Nrows
                table = table[0]
            tables.append(table)
        except Exception, e:
            print "ERROR:", e
            # make empty table using the dtype of the previous table
            last = tables[-1] # XXX BUG: first query cannot be null
            tables.append(nulltable(last.dtype))
    result = astropy.table.vstack(tables)
    result.remove_column('INPUT') # query_logg also has this
    querycols = result.colnames
    result['ORDER'] = np.arange(len(obj_list))
    result['INPUT'] = obj_list
    neworder = ['ORDER', 'INPUT'] + querycols
    result = result[neworder]
    return result
