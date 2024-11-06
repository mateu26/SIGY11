using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using SIGY11.Components;
using System.Net.Http;

var builder = WebApplication.CreateBuilder(args);

// Register HttpClient service for DI in the Blazor Server app
builder.Services.AddHttpClient();

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseAntiforgery();

// Map Razor components
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
