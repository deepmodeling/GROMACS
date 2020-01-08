/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2018 by the GROMACS development team.
 * Copyright (c) 2019,2020, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#ifndef GMX_TOPOLOGY_IDEF_H
#define GMX_TOPOLOGY_IDEF_H

#include <cstdio>

#include <array>
#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

struct gmx_ffparams_t;

typedef union t_iparams {
    /* Some parameters have A and B values for free energy calculations.
     * The B values are not used for regular simulations of course.
     * Free Energy for nonbondeds can be computed by changing the atom type.
     * The harmonic type is used for all harmonic potentials:
     * bonds, angles and improper dihedrals
     */
    struct
    {
        real a, b, c;
    } bham;
    struct
    {
        real rA, krA, rB, krB;
    } harmonic;
    struct
    {
        real klinA, aA, klinB, aB;
    } linangle;
    struct
    {
        real lowA, up1A, up2A, kA, lowB, up1B, up2B, kB;
    } restraint;
    /* No free energy supported for cubic bonds, FENE, WPOL or cross terms */
    struct
    {
        real b0, kb, kcub;
    } cubic;
    struct
    {
        real bm, kb;
    } fene;
    struct
    {
        real r1e, r2e, krr;
    } cross_bb;
    struct
    {
        real r1e, r2e, r3e, krt;
    } cross_ba;
    struct
    {
        real thetaA, kthetaA, r13A, kUBA, thetaB, kthetaB, r13B, kUBB;
    } u_b;
    struct
    {
        real theta, c[5];
    } qangle;
    struct
    {
        real alpha;
    } polarize;
    struct
    {
        real alpha, drcut, khyp;
    } anharm_polarize;
    struct
    {
        real al_x, al_y, al_z, rOH, rHH, rOD;
    } wpol;
    struct
    {
        real a, alpha1, alpha2, rfac;
    } thole;
    struct
    {
        real c6, c12;
    } lj;
    struct
    {
        real c6A, c12A, c6B, c12B;
    } lj14;
    struct
    {
        real fqq, qi, qj, c6, c12;
    } ljc14;
    struct
    {
        real qi, qj, c6, c12;
    } ljcnb;
    /* Proper dihedrals can not have different multiplicity when
     * doing free energy calculations, because the potential would not
     * be periodic anymore.
     */
    struct
    {
        real phiA, cpA;
        int  mult;
        real phiB, cpB;
    } pdihs;
    struct
    {
        real dA, dB;
    } constr;
    /* Settle can not be used for Free energy calculations of water bond geometry.
     * Use shake (or lincs) instead if you have to change the water bonds.
     */
    struct
    {
        real doh, dhh;
    } settle;
    struct
    {
        real b0A, cbA, betaA, b0B, cbB, betaB;
    } morse;
    struct
    {
        real pos0A[DIM], fcA[DIM], pos0B[DIM], fcB[DIM];
    } posres;
    struct
    {
        real pos0[DIM], r, k;
        int  geom;
    } fbposres;
    struct
    {
        real rbcA[NR_RBDIHS], rbcB[NR_RBDIHS];
    } rbdihs;
    struct
    {
        real cbtcA[NR_CBTDIHS], cbtcB[NR_CBTDIHS];
    } cbtdihs;
    struct
    {
        real a, b, c, d, e, f;
    } vsite;
    struct
    {
        int  n;
        real a;
    } vsiten;
    /* NOTE: npair is only set after reading the tpx file */
    struct
    {
        real low, up1, up2, kfac;
        int  type, label, npair;
    } disres;
    struct
    {
        real phiA, dphiA, kfacA, phiB, dphiB, kfacB;
    } dihres;
    struct
    {
        int  ex, power, label;
        real c, obs, kfac;
    } orires;
    struct
    {
        int  table;
        real kA;
        real kB;
    } tab;
    struct
    {
        int cmapA, cmapB;
    } cmap;
    struct
    {
        real buf[MAXFORCEPARAM];
    } generic; /* Conversion */
} t_iparams;

typedef int t_functype;

//! Reference to an interaction list entry, used by the iterator
struct InteractionListEntry
{
    //! Constructor using pointer to raw indices and the stride of the interaction list
    InteractionListEntry(int const* indicesPtr, const std::size_t stride) :
        parameterType(indicesPtr[0]),
        atoms(indicesPtr + 1, indicesPtr + stride)
    {
    }

    //! Reference to the parameter type
    const int& parameterType;
    //! Reference to the atoms
    gmx::ArrayRef<const int> atoms;
};

//! Helper class for looping over entries in an InteractionList
class InteractionListIterator
{
public:
    //! Difference type for iterator arithmetic
    using difference_type = std::ptrdiff_t;

    //! Constructor
    InteractionListIterator(const int* indicesPtr, std::size_t stride) :
        indicesPtr_(indicesPtr),
        stride_(stride)
    {
    }
    //! Value
    operator InteractionListEntry() const { return InteractionListEntry(indicesPtr_, stride_); }
    //! Pointer
    InteractionListEntry operator*() const { return InteractionListEntry(indicesPtr_, stride_); }
    //! Equality comparison
    bool operator==(const InteractionListIterator other)
    {
        return indicesPtr_ == other.indicesPtr_;
    }
    //! Inequality comparison
    bool operator!=(const InteractionListIterator other)
    {
        return indicesPtr_ != other.indicesPtr_;
    }
    //! Increment operator
    InteractionListIterator& operator++()
    {
        indicesPtr_ += stride_;
        return *this;
    }
    //! Increment operator
    InteractionListIterator operator++(int gmx_unused dummy)
    {
        InteractionListIterator tmp(*this);
        indicesPtr_ += stride_;
        return tmp;
    }
    //! Addition operator
    InteractionListIterator operator+(difference_type d)
    {
        return { indicesPtr_ + d * stride_, stride_ };
    }

private:
    //! Pointer to the first index of the current entry
    int const* indicesPtr_;
    //! The stride of entries in the flat list
    std::size_t stride_;
};

//! Reference to an interaction list entry, used by the iterator
struct InteractionListWriteEntry
{
    //! Constructor using pointer to raw indices and the stride of entries in the interaction list
    InteractionListWriteEntry(int* indicesPtr, const std::size_t stride) :
        parameterType(indicesPtr[0]),
        atoms(indicesPtr + 1, indicesPtr + stride)
    {
    }

    //! Reference to the parameter type
    int& parameterType;
    //! Reference to the atoms
    gmx::ArrayRef<int> atoms;
};

//! Helper class for looping over entries in an InteractionList with write access to entries
class InteractionListWriteIterator
{
public:
    //! Difference type for iterator arithmetic
    using difference_type = std::ptrdiff_t;

    //! Constructor
    InteractionListWriteIterator(int* indicesPtr, const std::size_t stride) :
        indicesPtr_(indicesPtr),
        stride_(stride)
    {
    }
    //! Value
    operator InteractionListWriteEntry() const
    {
        return InteractionListWriteEntry(indicesPtr_, stride_);
    }
    //! Pointer
    InteractionListWriteEntry operator*() const
    {
        return InteractionListWriteEntry(indicesPtr_, stride_);
    }
    //! Equality comparison
    bool operator==(const InteractionListWriteIterator other)
    {
        return indicesPtr_ == other.indicesPtr_;
    }
    //! Inequality comparison
    bool operator!=(const InteractionListWriteIterator other)
    {
        return indicesPtr_ != other.indicesPtr_;
    }
    //! Increment operator
    InteractionListWriteIterator& operator++()
    {
        indicesPtr_ += stride_;
        return *this;
    }
    //! Increment operator
    InteractionListWriteIterator operator++(int gmx_unused dummy)
    {
        InteractionListWriteIterator tmp(*this);
        indicesPtr_ += stride_;
        return tmp;
    }
    //! Compound += operator
    InteractionListWriteIterator operator+=(difference_type d)
    {
        indicesPtr_ += d * stride_;
        return *this;
    }
    //! Addition operator
    InteractionListWriteIterator operator+(difference_type d)
    {
        return { indicesPtr_ + d * stride_, stride_ };
    }

private:
    //! Pointer to the first index of the current entry
    int* indicesPtr_;
    //! The stride of entries in the interaction list
    std::size_t stride_;
};

/*! \brief List of listed interactions
 *
 * Each interaction consists of a parameter type and \p numAtoms() atom indices.
 * Iterators are provided to iterate over the interactions.
 * For maximum performance in force kernels, direct access to the underlying
 * flat array or integers is provided.
 */
class InteractionList
{
public:
    //! Constructs an empty list
    InteractionList(int functionType) : functionType_(functionType), stride_(1 + NRAL(functionType))
    {
    }

    //! Constructs from raw indices stored as: t0,a00,a01,...,t1,a10,a11,...
    InteractionList(int functionType, std::vector<int>&& rawIndices) :
        functionType_(functionType),
        indices_(std::move(rawIndices)),
        stride_(1 + NRAL(functionType))
    {
        GMX_RELEASE_ASSERT(indices_.size() % stride_ == 0,
                           "The number of raw indices should be a multiple of 1 + the number of "
                           "atoms per interaction");
    }

    //! Returns the function type
    int functionType() const { return functionType_; }

    //! Returns the number of interactions in the list
    std::size_t numInteractions() const { return int(indices_.size()) / int(stride_); }

    //! Returns whether the list is empty
    bool empty() const { return indices_.empty(); }

    //! Returns the number of atoms in an interaction
    std::size_t numAtoms() const { return stride_ - 1; }

    //! Return an entry from the list
    const InteractionListEntry operator[](std::size_t interactionIndex) const
    {
        return InteractionListEntry(indices_.data() + interactionIndex * stride_, stride_);
    }

    //! Return the raw indices in layout: t0,a00,a01,...,t1,a10,a11,...
    gmx::ArrayRef<const int> rawIndices() const { return indices_; }

    //! Return the raw indices in layout: t0,a00,a01,...,t1,a10,a11,...
    gmx::ArrayRef<const int> iatoms() const { return indices_; }

    //! Adds an interaction at the end of the list
    void push_back(const int parameterType, const std::size_t numAtoms, const int* atoms)
    {
        GMX_ASSERT(numAtoms == stride_ - 1,
                   "The number of atoms in the interaction to add should match the number of atoms "
                   "of the function type of the list");
        const std::size_t oldSize = indices_.size();
        indices_.resize(indices_.size() + stride_);
        indices_[oldSize] = parameterType;
        for (std::size_t i = 1; i < stride_; i++)
        {
            indices_[oldSize + i] = atoms[i - 1];
        }
    }

    //! Adds an interaction at the end of the list
    template<std::size_t numAtoms>
    void push_back(const int parameterType, const std::array<int, numAtoms>& atoms)
    {
        push_back(parameterType, numAtoms, atoms.data());
    }

    //! Adds an interaction at the end of the list
    void push_back(const int parameterType, gmx::ArrayRef<const int> atoms)
    {
        push_back(parameterType, atoms.size(), atoms.data());
    }

    //! Appends \p ilist at the back of the list, the function types should match (except that F_CONSTRNC can be appended to F_CONSTR)
    void append(const InteractionList& ilist)
    {
        GMX_RELEASE_ASSERT(ilist.functionType() == functionType_
                                   || (functionType_ == F_CONSTR && ilist.functionType() == F_CONSTRNC),
                           "Function types should match");
        indices_.insert(indices_.end(), ilist.iatoms().begin(), ilist.iatoms().end());
    }

    //! Clears the list
    void clear() { indices_.clear(); }

    //! Returns the begin iterator
    inline InteractionListIterator begin() const
    {
        return InteractionListIterator(indices_.data(), stride_);
    }
    //! Returns the end iterator
    inline InteractionListIterator end() const
    {
        return InteractionListIterator(indices_.data() + indices_.size(), stride_);
    }

    //! Returns the begin iterator
    inline InteractionListWriteIterator begin()
    {
        return InteractionListWriteIterator(indices_.data(), stride_);
    }
    //! Returns the end iterator
    inline InteractionListWriteIterator end()
    {
        return InteractionListWriteIterator(indices_.data() + indices_.size(), stride_);
    }

private:
    //! The function type
    int functionType_;
    //! List of interactions, stored as t0,a00,a01,...,t1,a10,a11,...
    std::vector<int> indices_;
    //! The stride of entries in indices_, equal to 1 + numAtoms()
    std::size_t stride_;
};

//! Helper class for looping over selections of InteractionLists
class InteractionListsSelection
{
public:
    //! Iterator for loop over selected InteractionList entries in an InteractionLists
    class iterator
    {
    public:
        //! Difference type for iterator arithmetic
        using difference_type = std::ptrdiff_t;

        //! Constructor
        iterator(const InteractionList* listPtr, const InteractionList* listEnd, int flags) :
            listPtr_(listPtr),
            listEnd_(listEnd),
            flags_(flags)
        {
            findValidList();
        }
        //! Pointer
        const InteractionList& operator*() const { return *listPtr_; }
        //! Equality comparison
        bool operator==(const iterator other) { return listPtr_ == &(*other); }
        //! Inequality comparison
        bool operator!=(const iterator other) { return listPtr_ != &(*other); }
        //! Increment operator
        iterator& operator++()
        {
            listPtr_++;
            findValidList();
            return *this;
        }
        //! Increment operator
        iterator operator++(int gmx_unused dummy)
        {
            iterator tmp(listPtr_, listEnd_, flags_);
            (*this)++;
            return tmp;
        }

    private:
        //! Increases listPtr_ until a non-empty list is found with one of the flags set
        void findValidList()
        {
            while (listPtr_ < listEnd_
                   && ((interaction_function[listPtr_->functionType()].flags & flags_) == 0
                       || listPtr_->empty()))
            {
                listPtr_++;
            }
        }

        //! Pointer to the current list
        const InteractionList* listPtr_;
        //! Pointer to the one after the last list
        const InteractionList* listEnd_;
        //! The interaction flags to select on
        int flags_;
    };

    //! Constructor
    InteractionListsSelection(gmx::ArrayRef<const InteractionList> ilists, int flags) :
        ilists_(ilists),
        flags_(flags)
    {
    }

    //! Returns iterator to begin
    const iterator begin() const
    {
        return iterator(ilists_.data(), ilists_.data() + ilists_.size(), flags_);
    }

    //! Returns iterator to end
    const iterator end() const
    {
        return iterator(ilists_.data() + ilists_.size(), ilists_.data() + ilists_.size(), flags_);
    }

private:
    //! The interaction lists to select from
    gmx::ArrayRef<const InteractionList> ilists_;
    //! The interaction flags to select on
    int flags_;
};

/* List of interaction lists, one list for each interaction type
 *
 * TODO: Consider only including entries in use instead of all F_NRE
 */
class InteractionLists
{
public:
    InteractionLists()
    {
        for (int ftype = 0; ftype < F_NRE; ftype++)
        {
            lists_.emplace_back(ftype);
        }
    }

    std::size_t size() const { return lists_.size(); }

    InteractionList&       operator[](std::size_t listIndex) { return lists_[listIndex]; }
    const InteractionList& operator[](std::size_t listIndex) const { return lists_[listIndex]; }
    InteractionList&       at(std::size_t listIndex) { return lists_.at(listIndex); }
    const InteractionList& at(std::size_t listIndex) const { return lists_.at(listIndex); }

    const InteractionList* data() const { return lists_.data(); }

    //! Returns a selection of non-empty interaction lists that have one of the flags in \p flags set
    InteractionListsSelection selection(int flags) const
    {
        return InteractionListsSelection(lists_, flags);
    }

private:
    std::vector<InteractionList> lists_;
};

/* Deprecated list of listed interactions */
struct t_ilist
{
    /* Returns the total number of elements in iatoms */
    int size() const { return nr; }

    /* Returns whether the list is empty */
    bool empty() const { return nr == 0; }

    gmx::ArrayRef<const int> rawIndices() const { return gmx::constArrayRefFromArray(iatoms, nr); }

    int* iatomsWriteAccess() { return iatoms; }

    int      nr;
    t_iatom* iatoms;
    int      nalloc;
};

/* TODO: Remove t_ilist and remove templating on list type in mshift.cpp */

/*
 * The structs InteractionList and t_ilist defines a list of atoms with their interactions.
 * General field description:
 *   int nr
 *      the size (nr elements) of the interactions array (iatoms[]).
 *   t_iatom *iatoms
 *  specifies which atoms are involved in an interaction of a certain
 *       type. The layout of this array is as follows:
 *
 *        +-----+---+---+---+-----+---+---+-----+---+---+---+-----+---+---+...
 *        |type1|at1|at2|at3|type2|at1|at2|type1|at1|at2|at3|type3|at1|at2|
 *        +-----+---+---+---+-----+---+---+-----+---+---+---+-----+---+---+...
 *
 *  So for interaction type type1 3 atoms are needed, and for type2 and
 *      type3 only 2. The type identifier is used to select the function to
 *      calculate the interaction and its actual parameters. This type
 *      identifier is an index in a params[] and functype[] array.
 */

/* The following four iterator generators should be removed when t_ilist is removed */

//! Returns the begin iterator
inline InteractionListWriteIterator begin(InteractionList* iList, std::size_t numAtoms)
{
    GMX_ASSERT(numAtoms == iList->numAtoms(), "numAtoms should match");
    return iList->begin();
}
//! Returns the end iterator
inline InteractionListWriteIterator end(InteractionList* iList, std::size_t numAtoms)
{
    GMX_ASSERT(numAtoms == iList->numAtoms(), "numAtoms should match");
    return iList->end();
}

//! Returns the begin iterator
inline InteractionListWriteIterator begin(t_ilist* iList, std::size_t numAtoms)
{
    return InteractionListWriteIterator(iList->iatoms, numAtoms);
}
//! Returns the end iterator
inline InteractionListWriteIterator end(t_ilist* iList, std::size_t numAtoms)
{
    return InteractionListWriteIterator(iList->iatoms, numAtoms);
}

struct gmx_cmapdata_t
{
    std::vector<real> cmap; /* Has length 4*grid_spacing*grid_spacing, */
    /* there are 4 entries for each cmap type (V,dVdx,dVdy,d2dVdxdy) */
};

struct gmx_cmap_t
{
    int                         grid_spacing = 0; /* Grid spacing */
    std::vector<gmx_cmapdata_t> cmapdata; /* Lists of grids with actual, pre-interpolated data */
};


enum
{
    ilsortUNKNOWN,
    ilsortNO_FE,
    ilsortFE_UNSORTED,
    ilsortFE_SORTED
};

/* Struct with list of interaction parameters and lists of interactions
 *
 * TODO: Convert to a proper class with private data members so we can
 * ensure that the free-energy sorting and sorting setting is consistent.
 */
class InteractionDefinitions
{
public:
    /* Constructor
     *
     * \param[in] ffparams  The interaction parameters, the lifetime of the created object should not exceed the lifetime of the passed parameters
     */
    InteractionDefinitions(const gmx_ffparams_t& ffparams);

    // Clears data not read in from ffparams
    void clear();

    // The interaction parameters
    const std::vector<t_iparams>& iparams;
    // The function type per type
    const std::vector<int>& functype;
    // Position restraint interaction parameters
    std::vector<t_iparams> iparams_posres;
    // Flat-bottomed position restraint parameters
    std::vector<t_iparams> iparams_fbposres;
    // The list of interactions for each type. Note that some, such as LJ and COUL will have 0 entries.
    InteractionLists il;
    /* The number of non-perturbed interactions at the start of each entry in il */
    std::array<int, F_NRE> numNonperturbedInteractions;
    // The sorting state of interaction in il
    int ilsort = ilsortUNKNOWN;
    // The dihedral correction maps
    gmx_cmap_t cmap_grid;
};

/* Deprecated interation definitions, used in t_topology */
struct t_idef
{
    int         ntypes;
    int         atnr;
    t_functype* functype;
    t_iparams*  iparams;
    real        fudgeQQ;
    t_iparams * iparams_posres, *iparams_fbposres;

    t_ilist il[F_NRE];
    int     ilsort;
};

/*
 * The struct t_idef defines all the interactions for the complete
 * simulation. The structure is setup in such a way that the multinode
 * version of the program  can use it as easy as the single node version.
 * General field description:
 *   int ntypes
 *      defines the number of elements in functype[] and param[].
 *   int nodeid
 *      the node id (if parallel machines)
 *   int atnr
 *      the number of atomtypes
 *   t_functype *functype
 *      array of length ntypes, defines for every force type what type of
 *      function to use. Every "bond" with the same function but different
 *      force parameters is a different force type. The type identifier in the
 *      forceatoms[] array is an index in this array.
 *   t_iparams *iparams
 *      array of length ntypes, defines the parameters for every interaction
 *      type. The type identifier in the actual interaction list
 *      (ilist[ftype].iatoms[]) is an index in this array.
 *   gmx_cmap_t cmap_grid
 *      the grid for the dihedral pair correction maps.
 *   t_iparams *iparams_posres, *iparams_fbposres
 *      defines the parameters for position restraints only.
 *      Position restraints are the only interactions that have different
 *      parameters (reference positions) for different molecules
 *      of the same type. ilist[F_POSRES].iatoms[] is an index in this array.
 *   t_ilist il[F_NRE]
 *      The list of interactions for each type. Note that some,
 *      such as LJ and COUL will have 0 entries.
 *   int ilsort
 *      The state of the sorting of il, values are provided above.
 */

namespace gmx
{
class TextWriter;
} // namespace gmx

void printInteractionParameters(gmx::TextWriter* writer, t_functype ftype, const t_iparams& iparams);
void pr_iparams(FILE* fp, t_functype ftype, const t_iparams& iparams);
void pr_ilist(FILE*                  fp,
              int                    indent,
              const char*            title,
              const t_functype*      functype,
              const InteractionList& ilist,
              gmx_bool               bShowNumbers,
              gmx_bool               bShowParameters,
              const t_iparams*       iparams);
void pr_idef(FILE* fp, int indent, const char* title, const t_idef* idef, gmx_bool bShowNumbers, gmx_bool bShowParameters);

/*! \brief
 * Properly initialize idef struct.
 *
 * \param[in] idef Pointer to idef struct to initialize.
 */
void init_idef(t_idef* idef);

/*! \brief
 * Properly clean up idef struct.
 *
 * \param[in] idef Pointer to idef struct to clean up.
 */
void done_idef(t_idef* idef);

#endif
