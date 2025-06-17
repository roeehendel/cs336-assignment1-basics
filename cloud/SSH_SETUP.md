# GitHub SSH Setup for CS336 VMs

This guide helps you set up SSH access to GitHub on your VMs so you can push/pull without entering credentials.

## Quick Setup

1. **Run the setup script** (from your local machine):
   ```bash
   cd cloud
   ./setup_github_ssh.sh setup
   ```

2. **Add the public key to GitHub**:
   - The script will show you the public key to copy
   - Go to https://github.com/settings/ssh/new
   - Paste the key and give it a name like "CS336 VMs"

3. **Start using SSH**:
   - New VMs will automatically use SSH for cloning if the key exists
   - On existing VMs, change the remote URL:
     ```bash
     cd ~/assignment1-basics
     git remote set-url origin git@github.com:roeehendel/assignment1-basics.git
     ```

## Commands

- `./setup_github_ssh.sh setup` - Full setup (generate key + deploy to running VMs)
- `./setup_github_ssh.sh deploy` - Deploy existing key to running VMs  
- `./setup_github_ssh.sh status` - Check SSH setup on VMs
- `./setup_github_ssh.sh generate` - Only generate SSH key locally

## How it Works

1. **Local Key Generation**: Creates an SSH key on your local machine (`~/.ssh/cs336-github`)
2. **Key Deployment**: Copies the key to all running VMs
3. **Automatic Setup**: VMs automatically use SSH if the key is present

## Security Notes

- The SSH key is shared across all your VMs
- Keys are stored securely with proper permissions (600 for private, 644 for public)
- GitHub is automatically added to known_hosts on VMs
- You only need to add the public key to GitHub once

## Troubleshooting

If SSH isn't working:
1. Check key exists: `ls -la ~/.ssh/id_ed25519*` (on VM)
2. Test connection: `ssh -T git@github.com` (on VM)  
3. Check remote URL: `git remote -v` (should show `git@github.com:...`) 