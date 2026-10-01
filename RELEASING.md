# Publishing this package on GitHub and archiving it on Zenodo

The end state: a public GitHub repository, a `v1.0.0` release, and a Zenodo
record with a DOI that the paper can cite. Zenodo archives a GitHub repository
automatically, but only for releases published *after* the repository has been
switched on in Zenodo, so the order of the steps below matters.

Throughout, replace `<user>` with your GitHub account or organisation.

## 0. Before the first release

Whatever is in the tree when the release is cut is what gets archived forever,
so settle the metadata first.

1. Add the ORCID identifiers to `CITATION.cff`, under each author:

   ```yaml
   authors:
     - family-names: Huang
       given-names: Peiqi
       orcid: "https://orcid.org/0000-0000-0000-0000"
   ```

   and to `.zenodo.json`, as a field of each creator:

   ```json
   {"name": "Huang, Peiqi", "affiliation": "...", "orcid": "0000-0000-0000-0000"}
   ```

   Note the different conventions: `CITATION.cff` wants the full URL, Zenodo
   wants the bare identifier.

2. Check the year on the copyright line of `LICENSE` and the `version` fields
   of `CITATION.cff` and `.zenodo.json`.

3. Run the suite once more and commit:

   ```sh
   python -m pytest tests -q
   git add -A && git commit -m "Metadata for the first release"
   ```

## 1. Create the GitHub repository

The package already is a git repository with its history, so create an **empty**
repository on GitHub — no README, no licence, no `.gitignore`, or the first
push will be rejected as a non-fast-forward.

On <https://github.com/new>: name it `unfitted-hho-elastodynamics`, set the
description to *Unfitted HHO for elastodynamics with an imperfect interface:
verification and reproduction code*, and make it **public**. Zenodo cannot
archive a private repository.

Then push:

```sh
cd unfitted-hho-elastodynamics
git remote add origin https://github.com/<user>/unfitted-hho-elastodynamics.git
git branch -M main
git push -u origin main
```

With the `gh` command-line tool the two steps collapse into one:

```sh
gh repo create <user>/unfitted-hho-elastodynamics --public --source=. --push \
   --description "Unfitted HHO for elastodynamics with an imperfect interface"
```

Check on the repository page that the README renders, that GitHub has detected
the BSD 3-Clause licence in the sidebar, and that a *Cite this repository* button has
appeared — that button is GitHub reading `CITATION.cff`, and it failing to
appear means the file has a syntax error.

## 2. Switch the repository on in Zenodo

1. Go to <https://zenodo.org> and log in. Signing in with GitHub links the two
   accounts in one step; if you already have a Zenodo account, link GitHub
   instead under the account settings, *Linked accounts*.
2. Open the GitHub page of your Zenodo settings,
   <https://zenodo.org/account/settings/github/>. It lists the public
   repositories you can administer. If the new one is not there yet, use
   *Sync now* — the list is cached.
3. Flip the toggle next to `unfitted-hho-elastodynamics` to **On**.

From now on Zenodo receives a notification for every release of that
repository. Releases published before the toggle was on are **not** archived
retroactively.

## 3. Cut the release

On GitHub: *Releases* → *Draft a new release*.

* Tag: `v1.0.0`, created on publish, targeting `main`.
* Title: `v1.0.0 — code accompanying the paper`.
* Description: a few lines, for instance what the package contains and which
  paper it belongs to. This text becomes the Zenodo record's description only
  if `.zenodo.json` is absent; here `.zenodo.json` wins, so keep it short.
* Publish release.

Equivalently:

```sh
git tag -a v1.0.0 -m "Code accompanying the paper"
git push origin v1.0.0
gh release create v1.0.0 --title "v1.0.0 — code accompanying the paper" \
   --notes "Solver, production runs, raw results and verification suite."
```

## 4. Collect the DOI

Within a few minutes Zenodo downloads the repository archive, reads
`.zenodo.json` for the metadata, and publishes a record. Watch for it at
<https://zenodo.org/account/settings/github/> — the entry gains a DOI badge —
or under *My dashboard* → *Uploads*.

Zenodo mints **two** DOIs:

* a **concept DOI**, which always resolves to the newest version;
* a **version DOI**, specific to `v1.0.0`.

Cite the concept DOI in the paper unless you deliberately want to pin readers
to this exact version. The record page shows both; the concept DOI is the one
labelled *all versions*.

If the record does not appear, the usual causes are the toggle having been
switched on after the release was published, or the repository being private.
Fix the cause, then publish a `v1.0.1` release — Zenodo will archive that one.

## 5. Put the DOI back into the package and the paper

1. Add the badge at the top of `README.md`:

   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```

2. Add the DOI to `CITATION.cff`, at the top level:

   ```yaml
   doi: 10.5281/zenodo.XXXXXXX
   repository-code: "https://github.com/<user>/unfitted-hho-elastodynamics"
   ```

3. Commit and push. Whether to cut a `v1.0.1` release just to carry the badge
   inside the archive is a matter of taste; most people do not.

4. Add a data-availability statement to the manuscript, next to the
   acknowledgements:

   > The code reproducing all numerical experiments of this paper is available
   > at <https://github.com/\<user\>/unfitted-hho-elastodynamics> and archived at
   > <https://doi.org/10.5281/zenodo.XXXXXXX>.

## Archiving without GitHub

If you would rather not use GitHub at all, upload the zip directly:
<https://zenodo.org/uploads/new> → drag the archive in → *Upload type:
Software* → fill in title, authors, affiliations, ORCIDs, licence (BSD 3-Clause),
keywords → *Publish*. The DOI is minted on publication.

In a draft you can also press *Reserve DOI* to obtain the identifier before
publishing, which is what to do if the DOI has to appear in the manuscript you
are about to submit. The reserved DOI only becomes resolvable once you publish
the record, so publish before the paper goes to press.

## Afterwards: new versions

To release an update, commit the changes, bump `version` in `CITATION.cff` and
`.zenodo.json`, and publish a new GitHub release with a new tag. Zenodo adds it
as a new version under the same concept DOI, and the citation in the paper
keeps working.

---

Zenodo's interface is revised from time to time; if a page is not where this
file says it is, <https://help.zenodo.org> has the current walkthrough under
*GitHub integration*.
