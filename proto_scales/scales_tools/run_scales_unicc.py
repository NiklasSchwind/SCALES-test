import paramiko

hostname = "slurm-login.iiasa.ac.at"
username = "kainverena"

# script = """
# print("Hello from remote!")
# for i in range(5):
#     print(i)
# """

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname, username=username,port=30222)

inner = (
    "module load Python/3.11.5-GCCcore-13.2.0 && "
    "source /hdrive/all_users/kainverena/PythonProjects/venv_scales_mesh/bin/activate && "
    "python ~/PythonProjects/SCALES-test/proto_scales/scales_tools/_scales_cnp_fastmip.py"
)
cmd = f"bash -l -c '{inner}'"
stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)

# Block until the command actually finishes
exit_code = stdout.channel.recv_exit_status()

out = stdout.read().decode()
err = stderr.read().decode()

print("EXIT CODE:", exit_code)
print("STDOUT:", out)
print("STDERR:", err)        # this is where errors/tracebacks will appear

# sftp = ssh.open_sftp()

# remote_file = "/tmp/temp_script.py"

# with sftp.file(remote_file, "w") as f:
#     f.write(script)

# channel = ssh.invoke_shell()

# channel.send("module load Python/3.11.5-GCCcore-13.2.0\n")
# channel.send("source /hdrive/all_users/kainverena/PythonProjects/venv_scales_mesh/bin/activate\n")

#stdin, stdout, stderr = ssh.exec_command(f"python3 {remote_file}")
#stdin, stdout, stderr = ssh.exec_command(f"python ~/PythonProjects/SCALES-test/proto_scales/scales_tools/_scales_cnp_fastmip.py")

#channel.send("python ~/PythonProjects/SCALES-test/proto_scales/scales_tools/_scales_cnp_fastmip.py\n")

# while True:
#     if channel.recv_ready():
#         print(channel.recv(4096).decode(), end="")

#print(stdout.read().decode())

#ssh.exec_command(f"rm {remote_file}")

#sftp.close()
ssh.close()