import { test, expect } from '@playwright/test';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';

for (const width of [1440,375]) test(`account memory settings and conversation at ${width}px`,async({browser})=>{
  test.setTimeout(60000);
  const reservation=createServer();
  await new Promise<void>(r=>reservation.listen(0,'127.0.0.1',r));
  const address=reservation.address(); if(!address || typeof address==='string')throw Error('No port');
  const port=address.port; await new Promise<void>(r=>reservation.close(()=>r()));
  const origin=`http://127.0.0.1:${port}`;
  const child=spawn('D:/conda/python.exe',['tests/fixtures/memory_server.py',String(port),resolve('test-results','memory-'+randomUUID())],{cwd:'..',stdio:'ignore'});
  const context=await browser.newContext({viewport:{width,height:900}});
  try {
    await expect.poll(async()=>{try{return (await fetch(origin+'/fixture')).status;}catch{return 0;}},{timeout:20000}).toBe(200);
    const page=await context.newPage(); await page.goto(origin);
    await page.getByLabel('用户名').fill('demo_ab'); await page.getByLabel('密码').fill('MemoryTest123!');
    await page.getByRole('button',{name:'登录',exact:true}).click();
    async function openMemory(){
      if(width<600)await page.getByRole('button',{name:'打开对话列表'}).click();
      await page.getByRole('button',{name:'长期记忆',exact:true}).last().click();
      await expect(page.getByLabel('核心记忆')).toBeEnabled();
    }
    await openMemory();
    await page.getByLabel('核心记忆').fill('先给结论');
    await page.getByLabel('扩展记忆').fill('Python初学者\n\n劳动法初学者');
    await page.getByRole('button',{name:'保存记忆'}).click(); await expect(page.getByText('记忆已保存')).toBeVisible();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
    await page.screenshot({path:`test-results/memory-${width}.png`});
    await page.getByRole('button',{name:'关闭记忆设置'}).click();
    async function ask(text:string){await page.getByLabel('法律问题').fill(text);await page.getByRole('button',{name:'发送问题'}).click();}
    await ask('函数为什么报错');
    const first=page.locator('article.assistant').last();
    await expect(first).toContainText('Python初学者'); await expect(first).not.toContainText('劳动法初学者');
    await first.getByText('参考来源',{exact:false}).click(); await expect(first).toContainText('隔离测试法律资料');
    await expect(page.getByLabel('法律问题')).toBeEnabled();
    await ask('以后回答详细解释');
    await expect(page.getByText('已更新核心记忆。',{exact:true})).toBeVisible();
    await page.reload(); await openMemory();
    await expect(page.getByLabel('核心记忆')).toHaveValue('先给结论\n\n详细解释');
    await expect(page.getByLabel('扩展记忆')).toHaveValue('Python初学者\n\n劳动法初学者');
    await page.getByRole('button',{name:'关闭记忆设置'}).click();
    await fetch(origin+'/fixture/fail',{method:'POST'});
    await ask('劳动问题'); await expect(page.getByText('本次未使用扩展记忆',{exact:true})).toBeVisible();
    const last=page.locator('article.assistant').last(); await expect(last).toContainText('详细解释');await expect(last).not.toContainText('本次未使用扩展记忆');
    await openMemory(); await page.getByLabel('在回答中使用记忆').uncheck();
    await page.getByRole('button',{name:'保存记忆'}).click();await expect(page.getByText('记忆已保存')).toBeVisible();
    await page.getByRole('button',{name:'关闭记忆设置'}).click();
    await expect(page.getByLabel('法律问题')).toBeEnabled(); await ask('另一法律问题');
    await expect(page.locator('article.assistant').last()).toContainText('依据测试法条回答');
    await expect(page.getByLabel('法律问题')).toBeEnabled(); await expect(page.locator('article.assistant').last()).not.toContainText('详细解释');
    await expect(page.getByText('本次未使用扩展记忆',{exact:true})).toHaveCount(0);
    const colleague=await browser.newContext(); const other=await colleague.newPage();
    await other.goto(origin); await other.getByLabel('用户名').fill('demo_c');await other.getByLabel('密码').fill('MemoryTest123!');
    await other.getByRole('button',{name:'登录',exact:true}).click();await other.getByRole('button',{name:'长期记忆',exact:true}).click();
    await expect(other.getByLabel('核心记忆')).toHaveValue('');await expect(other.getByLabel('扩展记忆')).toHaveValue('');
    await colleague.close();
  } finally {
    await context.close();
    await fetch(origin+'/fixture/release',{method:'POST'}).catch(()=>{});
    if(child.exitCode===null){
      const exit=new Promise<void>(r=>child.once('exit',()=>r()));
      await fetch(origin+'/fixture/stop',{method:'POST'}).catch(()=>{});
      await Promise.race([exit,new Promise<void>(r=>setTimeout(r,5000))]);
      if(child.exitCode===null){child.kill();await exit;}
    }
  }
});
