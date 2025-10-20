# GitHub Authentication Setup Guide

**Repository:** github.com/astoreyai/falsifiable_attribution_data
**Status:** Ready to push (authentication required)
**Repository Size:** 95 MB (tracked files only)
**Files to Push:** 400 tracked files

---

## Quick Start (Choose ONE Method)

### ⭐ METHOD 1: Personal Access Token (FASTEST - 5 minutes)

```bash
# 1. Create token at: https://github.com/settings/tokens
#    - Click "Generate new token (classic)"
#    - Name: dissertation-backup
#    - Expiration: 90 days
#    - Scopes: ✅ repo (full control)
#    - Click "Generate token" and COPY IT

# 2. Configure git to store credentials
cd /home/aaron/projects/xai
git config --global credential.helper store

# 3. Push (you'll be prompted ONCE)
git push -u origin main

# When prompted:
# Username: astoreyai
# Password: [PASTE YOUR TOKEN - NOT your GitHub password!]

# Done! The token is saved, you won't need it again.
```

---

### METHOD 2: SSH Key (Most Secure - 10 minutes)

```bash
# 1. Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "astoreyai@github.com"
# Press Enter for defaults, optionally set passphrase

# 2. Copy public key
cat ~/.ssh/id_ed25519.pub
# Copy the entire output

# 3. Add to GitHub
#    - Go to: https://github.com/settings/keys
#    - Click "New SSH key"
#    - Title: dissertation-workstation
#    - Paste key
#    - Click "Add SSH key"

# 4. Switch remote to SSH
cd /home/aaron/projects/xai
git remote set-url origin git@github.com:astoreyai/falsifiable_attribution_data.git

# 5. Test connection
ssh -T git@github.com
# Should say: "Hi astoreyai! You've successfully authenticated..."

# 6. Push
git push -u origin main
```

---

### METHOD 3: GitHub CLI (Most Convenient - 7 minutes)

```bash
# 1. Install GitHub CLI (if not installed)
gh --version
# If not found: sudo apt install gh

# 2. Authenticate
gh auth login
# Choose:
#   - Account: GitHub.com
#   - Protocol: HTTPS
#   - Authenticate: Login with web browser
# Follow prompts

# 3. Push
cd /home/aaron/projects/xai
git push -u origin main
```

---

## What Will Be Pushed

### Included (400 tracked files, 95 MB)
- ✅ All source code (src/)
- ✅ All experiment scripts (experiments/*.py)
- ✅ Key figures (PDF format, ~40 files)
- ✅ Dissertation LaTeX (PHD_PIPELINE/falsifiable_attribution_dissertation/)
- ✅ Documentation (*.md files)
- ✅ Configuration files
- ✅ Requirements and setup files

### Excluded by .gitignore
- ❌ 2,188 PNG visualizations (~150 MB) - can be regenerated
- ❌ Virtual environments (venv/)
- ❌ Python cache (__pycache__/)
- ❌ LaTeX build artifacts (*.aux, *.log, etc.)
- ❌ Dataset downloads (data/lfw/, data/celeba/, etc.)
- ❌ Model weights (*.pt, *.pth)
- ❌ Large data files (*.csv, *.pkl, *.h5)

**Why exclude these?**
- Can be regenerated from source code
- Too large for GitHub (avoid LFS costs)
- Not needed for dissertation backup
- Datasets can be re-downloaded

---

## Verification After Push

Once push succeeds, verify:

```bash
cd /home/aaron/projects/xai

# Check tracking status
git branch -vv
# Should show: * main 1ab1d2e [origin/main] docs: Chapter 8 Writing Report

# Verify remote
git remote show origin
# Should show: main pushes to main (up to date)

# View on GitHub
# Go to: https://github.com/astoreyai/falsifiable_attribution_data
```

---

## Optional: Commit Recent Documentation

You have uncommitted files that could also be pushed:

```bash
cd /home/aaron/projects/xai

# Stage new documentation
git add COMPREHENSIVE_STATUS_REPORT.md
git add GIT_PUSH_INSTRUCTIONS.md
git add ORCHESTRATOR_FINAL_REPORT.md
git add ORCHESTRATOR_PROGRESS_LOG.md
git add SCENARIO_C_EXECUTION_PLAN_UPDATED.md
git add GIT_PUSH_SUMMARY.md
git add AUTHENTICATION_SETUP_GUIDE.md
git add defense/

# Commit
git commit -m "docs: Add orchestrator reports and authentication guides"

# Push
git push origin main
```

---

## Security Checklist

After successful push:

1. ✅ **Verify repository is PRIVATE**
   - Go to: https://github.com/astoreyai/falsifiable_attribution_data/settings
   - Check "Danger Zone" → "Change repository visibility"
   - Should say: **Private** (not Public)

2. ✅ **Verify .gitignore is working**
   - Check GitHub web interface
   - Should NOT see: venv/, __pycache__/, *.png visualizations

3. ✅ **Secure your credentials**
   - Personal Access Token: Store securely (password manager)
   - SSH Key: Use passphrase if possible
   - GitHub CLI: Uses OAuth (most secure)

---

## Troubleshooting

### Error: "Repository not found"
**Cause:** Repository doesn't exist on GitHub yet.

**Solution:**
1. Go to: https://github.com/new
2. Repository name: `falsifiable_attribution_data`
3. Visibility: **Private**
4. Do NOT initialize with README
5. Click "Create repository"
6. Then: `git push -u origin main`

---

### Error: "Permission denied (publickey)"
**Cause:** SSH key not configured or not added to GitHub.

**Solution:**
- Follow METHOD 2 steps above
- Ensure key is added to: https://github.com/settings/keys
- Test with: `ssh -T git@github.com`

---

### Error: "Authentication failed"
**Cause:** Wrong token or password.

**Solution:**
- Don't use your GitHub password - use a Personal Access Token
- Ensure token has `repo` scope
- Generate new token if needed: https://github.com/settings/tokens

---

### Warning: "This exceeds GitHub's file size limit of 100 MB"
**Cause:** Individual file larger than 100 MB.

**Solution:**
1. Check file size: `find . -type f -size +100M`
2. Add to .gitignore if not essential
3. Or use Git LFS (Large File Storage)

---

## Next Steps After Successful Push

1. 📱 **Set up mobile access**
   - Download GitHub mobile app
   - View your dissertation on the go

2. 🔄 **Regular backups**
   - After each work session: `git add . && git commit -m "work session" && git push`
   - Or use automation script

3. 🌿 **Branch protection** (optional)
   - Settings → Branches → Add branch protection rule
   - Protect `main` from force pushes

4. 👥 **Add collaborators** (if needed)
   - Settings → Collaborators → Add people
   - Your advisor (read-only access)

5. 📊 **GitHub Actions** (optional)
   - Auto-build LaTeX on every push
   - Run tests automatically

---

## Time Estimates

| Method | Setup Time | Difficulty | Best For |
|--------|-----------|-----------|----------|
| Personal Access Token | 5 min | Easy | Quick setup, temporary access |
| SSH Key | 10 min | Medium | Long-term use, most secure |
| GitHub CLI | 7 min | Easy | Frequent GitHub users |

---

## Support

If you encounter issues:

1. Read error message carefully
2. Check this guide's troubleshooting section
3. Verify authentication method is correct
4. Test with: `git remote -v` and `git status`
5. GitHub Docs: https://docs.github.com/en/authentication

---

**Ready to push?** Choose a method above and follow the steps! 🚀
